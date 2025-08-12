import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import sacrebleu
from acegen.models.gpt2 import GPT2, define_gpt2_configuration
from acegen.vocabulary.tokenizers import SMILESTokenizerEnamine


def compute_bleu(pred: str, ref: str) -> float:
    return sacrebleu.sentence_bleu(pred, [ref]).score


class SteeringVectorExtractor:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: SMILESTokenizerEnamine,
        vocab_path: str,
        d_prime: int = None,
        inject_layer: int = 6,
        inject_timestep: str = 'all',
        inject_mlp: bool = True,
        inject_every_layer: bool = False,   # match the “good” script
        inject_embedding: bool = False,     # match the “good” script
        device: torch.device = None,
        ckpt_path: str | None = None,
        steer_scale: float = 0.1,           # normalized injection scale
        reg_lambda: float = 5e-4,           # L2 on z (slightly lower)
        use_scheduler: bool = False,        # NEW: optional cosine scheduler
        clip_grad_norm: float | None = None # NEW: optional z grad clipping
    ):
        # Device
        self.device = device or (
            torch.device('cuda') if torch.cuda.is_available() else
            torch.device('mps') if torch.backends.mps.is_available() else
            torch.device('cpu')
        )

        # Load vocabulary
        if not os.path.isfile(vocab_path):
            raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
        with open(vocab_path, 'r') as vf:
            tokens = [line.strip() for line in vf if line.strip()]
        self.token2id = {tok: i for i, tok in enumerate(tokens)}
        self.id2token = {i: tok for tok, i in self.token2id.items()}
        vocab_size = len(tokens)

        # Model
        cfg = define_gpt2_configuration(vocabulary_size=vocab_size)
        self.model = (model or GPT2(cfg)).to(self.device)

        # HF core unwrap
        hf = self.model.feature_extractor
        if hasattr(hf.config, 'eos_token_id'):
            hf.config.pad_token_id = hf.config.eos_token_id
        self.hf_core = self.model.feature_extractor   # transformers.GPT2Model
        self.core = self.hf_core.transformer if hasattr(self.hf_core, "transformer") else self.hf_core
        self.layers = list(self.core.h)
        self.embedding_layer = self.core.wte

        # Output projection (default: tie to wte)
        self.output_weight = self.embedding_layer.weight  # [V, D]
        self.output_bias = None                           # optional lm_head.bias

        # Load checkpoint BEFORE freezing
        if ckpt_path is not None:
            self._smart_load_pretrained(ckpt_path)

        # Freeze model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # Tokenizer
        self.tokenizer = tokenizer or SMILESTokenizerEnamine()
        if hasattr(self.tokenizer, 'end_token'):
            self.tokenizer.pad_token = self.tokenizer.end_token

        # Dims & optional projection for z
        self.d = self.embedding_layer.weight.size(1)
        self.d_prime = d_prime or self.d
        self.W_steer = nn.Linear(self.d_prime, self.d, bias=False).to(self.device) if self.d_prime != self.d else None

        # Injection settings
        self.inject_layer = inject_layer
        self.inject_timestep = inject_timestep
        self.inject_mlp = inject_mlp
        self.inject_every_layer = inject_every_layer
        self.inject_embedding = inject_embedding

        # Steering/training controls
        self.steer_scale = steer_scale
        self.reg_lambda = reg_lambda
        self.use_scheduler = use_scheduler
        self.clip_grad_norm = clip_grad_norm

    # ----------  checkpoint loader (maps into AceGen wrapper; grabs lm_head if present) ----------
    # this is just to read the model weights from a generic checkpoint file and map them to the model
    def _smart_load_pretrained(self, ckpt_path: str):
        if not os.path.isfile(ckpt_path):
            print(f"[ckpt] file not found: {ckpt_path} (continuing with random init)")
            return

        sd = torch.load(ckpt_path, map_location=self.device)
        # unwrap nested dicts
        if isinstance(sd, dict):
            for key in ("state_dict", "model_state_dict", "weights"):
                if key in sd and isinstance(sd[key], dict):
                    sd = sd[key]
                    break

        # If it's a HuggingFace-like object with .state_dict()
        if hasattr(sd, "state_dict"):
            sd = sd.state_dict()

        target = self.model.state_dict()

        def strip_prefixes(k: str) -> str:
            for pref in ("model.", "module.", "prior.", "net."):
                if k.startswith(pref):
                    return k[len(pref):]
            return k

        def candidates_for(src_key: str) -> list[str]:
            """Generate plausible target keys for a given source key."""
            cands = set()
            k = strip_prefixes(src_key)

            # direct
            cands.add(k)

            # skip LM head for model mapping; handle separately
            if k.startswith("lm_head."):
                return []

            # transformer/gpt2 variations
            cands.add("feature_extractor." + k)
            cands.add("feature_extractor.transformer." + k)

            if k.startswith("transformer."):
                tail = k[len("transformer."):]
                cands.add("feature_extractor.transformer." + tail)
                cands.add("feature_extractor." + k)

            if k.startswith("gpt2."):
                tail = k[len("gpt2."):]
                cands.add("feature_extractor." + tail)
                cands.add("feature_extractor.transformer." + tail)

            # raw block pieces (wte/wpe/h/ln_f)
            starts = ("wte.", "wpe.", "h.", "ln_f.", "ln_", "attn.", "mlp.")
            for s in starts:
                idx = k.find(s)
                if idx >= 0:
                    tail = k[idx:]
                    cands.add("feature_extractor." + tail)
                    cands.add("feature_extractor.transformer." + tail)

            return list(cands)

        # Map model params by name & shape
        mapped = {}
        used = set()
        matched = 0
        for k, v in sd.items():
            # map model weights (not lm_head)
            for cand in candidates_for(k):
                if cand in target and target[cand].shape == v.shape and cand not in used:
                    mapped[cand] = v
                    used.add(cand)
                    matched += 1
                    break

        # Compose state dict and load
        new_state = target.copy()
        new_state.update(mapped)
        missing, unexpected = self.model.load_state_dict(new_state, strict=False)
        print(f"[ckpt] loaded '{ckpt_path}'  matched={matched}  missing={len(missing)}  unexpected={len(unexpected)}")

        # Try to capture a separate LM head weight/bias to use for projection
        lm_head_weight = None
        lm_head_bias = None

        # common head weight/bias keys to probe (with prefixes stripped)
        stripped = {strip_prefixes(k): k for k in sd.keys()}
        weight_keys = [
            "lm_head.weight", "head.weight", "output.weight", "decoder.weight", "lm_head.decoder.weight",
        ]
        bias_keys = [
            "lm_head.bias", "head.bias", "output.bias", "decoder.bias",
        ]

        for short in weight_keys:
            if short in stripped:
                lm_head_weight = sd[stripped[short]]
                break
        if lm_head_weight is None:
            # also allow suffix matches
            for k in sd.keys():
                ks = strip_prefixes(k)
                if ks.endswith("lm_head.weight") or ks.endswith("decoder.weight") or ks.endswith("output.weight"):
                    lm_head_weight = sd[k]
                    break

        for short in bias_keys:
            if short in stripped:
                lm_head_bias = sd[stripped[short]]
                break
        if lm_head_bias is None:
            for k in sd.keys():
                ks = strip_prefixes(k)
                if ks.endswith("lm_head.bias") or ks.endswith("decoder.bias") or ks.endswith("output.bias"):
                    lm_head_bias = sd[k]
                    break

        if lm_head_weight is not None:
            V, D = lm_head_weight.shape[0], lm_head_weight.shape[1]
            if V == self.embedding_layer.weight.shape[0] and D == self.embedding_layer.weight.shape[1]:
                self.output_weight = lm_head_weight.to(self.device)
                print(f"[ckpt] using separate lm_head.weight for logits: shape={tuple(self.output_weight.shape)}")
            else:
                print(f"[ckpt] head weight shape {tuple(lm_head_weight.shape)} != embedding {tuple(self.embedding_layer.weight.shape)}; using wte.T")

        if lm_head_bias is not None:
            if lm_head_bias.shape[0] == self.output_weight.shape[0]:
                self.output_bias = lm_head_bias.to(self.device)
                print(f"[ckpt] using lm_head.bias for logits: shape={tuple(self.output_bias.shape)}")
            else:
                print(f"[ckpt] head bias shape {tuple(lm_head_bias.shape)} != V={self.output_weight.shape[0]} — skipping.")

    # ---------- Tokenize ----------
    def _encode_smiles(self, smiles: str):
        base_toks = self.tokenizer.tokenize(smiles)
        toks = ['GO'] + base_toks + ['EOS']
        ids = [self.token2id.get(t, self.token2id.get('<pad>', 0)) for t in toks]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        seq_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.device)
        return input_ids, seq_mask

    # ---------- Projection helper (uses lm_head if available) ----------
    def _project(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: [B, T, D] or [B, D]; output logits [B, T, V] or [B, V]
        W = self.output_weight  # [V, D]
        if hidden.dim() == 3:
            logits = hidden @ W.t()  # [B, T, V]
        else:
            logits = hidden @ W.t()  # [B, V]
        if self.output_bias is not None:
            logits = logits + self.output_bias
        return logits

    # ---------- Injection (normalized + scaled) ----------
    def _inject(self, hidden: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        vec = self.W_steer(z) if self.W_steer is not None else z
        # normalize & scale to prevent runaway norms
        vec = vec / (vec.norm() + 1e-8)
        vec = self.steer_scale * vec
        if self.inject_timestep == 'first':
            hidden[:, 0, :] += vec
        else:
            hidden += vec.unsqueeze(0).unsqueeze(1)
        return hidden

    # ---------- Baseline LM loss (no steering) ----------
    def _baseline_loss(self, input_ids, seq_mask) -> float:
        with torch.no_grad():
            out = self.hf_core(input_ids=input_ids, attention_mask=seq_mask, return_dict=True)
            hidden_seq = out.last_hidden_state
            logits = self._project(hidden_seq)
            pad_id = self.token2id.get('<pad>', 0)
            loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:].clone()
            # don’t try to learn to predict GO
            go_id = self.token2id.get('GO', pad_id)
            shift_labels[shift_labels == go_id] = pad_id
            loss = loss_fn(shift_logits.reshape(-1, logits.size(-1)),
                           shift_labels.reshape(-1))
            return float(loss.item())

    # ---------- Training (teacher-forced full sequence) ----------
    def extract_zsteer(self, smiles: str, steps: int = 500, lr: float = 1.0) -> torch.Tensor:
        input_ids, seq_mask = self._encode_smiles(smiles)

        # quick baseline sanity
        base = self._baseline_loss(input_ids, seq_mask)
        print(f"[baseline LM loss (no steering)]: {base:.4f}")

        # init z
        zsteer = nn.Parameter(torch.empty(self.d_prime, device=self.device))
        nn.init.xavier_normal_(zsteer.unsqueeze(0))

        optimizer = torch.optim.Adam([zsteer], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps) if self.use_scheduler else None

        pad_id = self.token2id.get('<pad>', 0)
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
        go_id = self.token2id.get('GO', pad_id)

        # hooks
        handles = []
        if self.inject_embedding:
            handles.append(self.embedding_layer.register_forward_hook(lambda m, i, out: self._inject(out, zsteer)))
        target_layers = self.layers if self.inject_every_layer else [self.layers[self.inject_layer]]
        for lyr in target_layers:
            handles.append(
                lyr.attn.register_forward_hook(
                    lambda m, i, out, z=zsteer:
                    (self._inject(out[0], z), *out[1:]) if isinstance(out, tuple) else self._inject(out, z)
                )
            )
            if self.inject_mlp:
                handles.append(lyr.mlp.register_forward_hook(lambda m, i, out: self._inject(out, zsteer)))

        # train with HF core (get [B, T, D])
        self.model.train()
        for step in range(1, steps + 1):
            optimizer.zero_grad()

            out = self.hf_core(input_ids=input_ids, attention_mask=seq_mask, return_dict=True)
            hidden_seq = out.last_hidden_state                       # [B, T, D]
            logits = self._project(hidden_seq)                       # [B, T, V]

            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:].clone()
            # ignore loss on GO positions
            shift_labels[shift_labels == go_id] = pad_id

            loss = loss_fn(shift_logits.reshape(-1, logits.size(-1)),
                           shift_labels.reshape(-1))

            # L2 regularize z
            if self.reg_lambda > 0:
                loss = loss + self.reg_lambda * (zsteer.pow(2).sum())

            loss.backward()

            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_([zsteer], self.clip_grad_norm)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if step % max(1, steps // 10) == 0 or step == 1:
                cur_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
                print(f"Step {step}/{steps}  Loss: {loss.item():.4f}  Norm: {zsteer.norm().item():.2f}  LR: {cur_lr:.3e}")

        self.model.eval()
        for h in handles:
            h.remove()

        return zsteer.detach()

    # ---------- Helpers ----------
    @staticmethod
    def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
        if k is None or k <= 0:
            return logits
        # keep top-k, set others to -inf
        topk_vals, topk_idx = torch.topk(logits, k, dim=-1)
        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(dim=-1, index=topk_idx, src=topk_vals)
        return mask

    # ---------- Decoding  ----------
    def steer_generate(
        self,
        z: torch.Tensor,
        max_new_tokens: int = 20,
        steer_scale: float | None = None,
        min_decode_len: int = 0   
    ) -> str:
        # temporary override of steer strength at inference
        old_scale = self.steer_scale
        if steer_scale is not None:
            self.steer_scale = steer_scale
        try:
            z = z.to(self.device)
            pad_id = self.token2id.get('<pad>', 0)
            go_id  = self.token2id.get('GO', pad_id)
            eos_id = self.token2id.get('EOS', pad_id)

            hooks = []
            if self.inject_embedding:
                hooks.append(self.embedding_layer.register_forward_hook(lambda m, i, out: self._inject(out, z)))
            target_layers = self.layers if self.inject_every_layer else [self.layers[self.inject_layer]]
            for lyr in target_layers:
                hooks.append(
                    lyr.attn.register_forward_hook(
                        lambda m, i, out, z=z:
                        (self._inject(out[0], z), *out[1:]) if isinstance(out, tuple) else self._inject(out, z)
                    )
                )
                if self.inject_mlp:
                    hooks.append(lyr.mlp.register_forward_hook(lambda m, i, out: self._inject(out, z)))

            generated_ids = [go_id]
            for t in range(max_new_tokens):
                input_ids = torch.tensor([generated_ids], dtype=torch.long, device=self.device)
                seq_mask  = torch.ones_like(input_ids, device=self.device)

                out = self.hf_core(input_ids=input_ids, attention_mask=seq_mask, return_dict=True)
                hidden_last = out.last_hidden_state[:, -1, :]                 # [B, D]
                logits = self._project(hidden_last)                           # [B, V]

                # min-length guard: suppress EOS until min_decode_len reached
                if t < min_decode_len:
                    logits[:, eos_id] = float('-inf')

                next_id = int(logits.argmax(dim=-1).item())
                if next_id in (eos_id, pad_id):
                    break
                generated_ids.append(next_id)

            for h in hooks:
                h.remove()

            tokens = [self.id2token.get(i, '') for i in generated_ids if i not in (go_id, eos_id)]
            return self.tokenizer.untokenize(tokens)
        finally:
            self.steer_scale = old_scale  # restore

    def steer_transfer(
        self,
        z: torch.Tensor,
        prompt: str,
        max_new_tokens: int = 50,
        steer_scale: float | None = None,
        min_decode_len: int = 0,      # NEW
        temperature: float | None = None,  # NEW: only used if provided
        top_k: int | None = None            # NEW: only used if provided
    ) -> str:
        old_scale = self.steer_scale
        if steer_scale is not None:
            self.steer_scale = steer_scale
        try:
            z = z.to(self.device)
            pad_id = self.token2id.get('<pad>', 0)
            go_id  = self.token2id.get('GO', pad_id)
            eos_id = self.token2id.get('EOS', pad_id)

            hooks = []
            if self.inject_embedding:
                hooks.append(self.embedding_layer.register_forward_hook(lambda m, i, out: self._inject(out, z)))
            target_layers = self.layers if self.inject_every_layer else [self.layers[self.inject_layer]]
            for lyr in target_layers:
                hooks.append(
                    lyr.attn.register_forward_hook(
                        lambda m, i, out, z=z:
                        (self._inject(out[0], z), *out[1:]) if isinstance(out, tuple) else self._inject(out, z)
                    )
                )
                if self.inject_mlp:
                    hooks.append(lyr.mlp.register_forward_hook(lambda m, i, out: self._inject(out, z)))

            base_toks = self.tokenizer.tokenize(prompt)
            generated_ids = [go_id] + [self.token2id.get(t, pad_id) for t in base_toks]

            for t in range(max_new_tokens):
                input_ids = torch.tensor([generated_ids], dtype=torch.long, device=self.device)
                seq_mask  = torch.ones_like(input_ids, device=self.device)

                out = self.hf_core(input_ids=input_ids, attention_mask=seq_mask, return_dict=True)
                hidden_last = out.last_hidden_state[:, -1, :]
                logits = self._project(hidden_last)

                # min-length guard
                if t < min_decode_len:
                    logits[:, eos_id] = float('-inf')

                if temperature is not None or (top_k is not None and top_k > 0):
                    # sampling
                    temp = temperature if (temperature is not None and temperature > 0) else 1.0
                    logits = logits / temp
                    logits = self._top_k_filter(logits, top_k)
                    probs = torch.softmax(logits, dim=-1)
                    next_id = int(torch.multinomial(probs, num_samples=1).item())
                else:
                    # greedy
                    next_id = int(logits.argmax(dim=-1).item())

                if next_id in (eos_id, pad_id):
                    break
                generated_ids.append(next_id)

            for h in hooks:
                h.remove()

            tokens = [self.id2token.get(i, '') for i in generated_ids if i not in (go_id, eos_id)]
            return self.tokenizer.untokenize(tokens)
        finally:
            self.steer_scale = old_scale

    # ---------- Simple scale tuner for reconstruction ----------
    def tune_scale_for_reconstruction(self, z: torch.Tensor, target: str, scales=(0.05, 0.1, 0.2, 0.4, 0.8)):
        best_bleu, best_scale, best_text = -1.0, None, ""
        target_len = len(self.tokenizer.tokenize(target))
        for s in scales:
            txt = self.steer_generate(z, max_new_tokens=target_len, steer_scale=s)
            bleu = compute_bleu(txt, target)
            print(f"[tune] scale={s:.3f}  BLEU={bleu:.2f}")
            if bleu > best_bleu:
                best_bleu, best_scale, best_text = bleu, s, txt
        print(f"[tune] best scale={best_scale}  BLEU={best_bleu:.2f}")
        return best_scale, best_text, best_bleu


if __name__ == '__main__':
    vocab_file = 'acegen-open/acegen/priors/enamine_real_vocabulary.txt'
    tok = SMILESTokenizerEnamine()

    with open(vocab_file, 'r') as vf:
        vocab_size = sum(1 for _ in vf if _.strip())
    config = define_gpt2_configuration(vocabulary_size=vocab_size)
    model = GPT2(config)

    ckpt_path = 'acegen-open/acegen/priors/gpt2_enamine_real.ckpt'
    if not os.path.isfile(ckpt_path):
        print("[ckpt] No checkpoint file found — running with random weights (expect poor results).")

    extractor = SteeringVectorExtractor(
        model=model,
        tokenizer=tok,
        vocab_path=vocab_file,
        d_prime=768,
        inject_layer=6,
        inject_timestep='all',
        inject_mlp=True,
        inject_every_layer=False,   # single layer
        inject_embedding=False,     # no embedding hook
        ckpt_path=ckpt_path,
        steer_scale=0.1,
        reg_lambda=5e-4,
        use_scheduler=False,        # flip to True to try cosine anneal
        clip_grad_norm=None         # e.g. set to 1.0 to clamp updates
    )

    target = 'Nc1ncnc2c1ncn2C1OC(COP(=O)(O)OP(=O)(O)OC2OC(CO)C(O)C(O)C2O)C(O)C10'
    z = extractor.extract_zsteer(target, steps=500, lr=0.01)

    # vanilla recon (with min length to discourage early EOS)
    rec = extractor.steer_generate(
        z,
        max_new_tokens=len(tok.tokenize(target)),
        min_decode_len=max(0, len(tok.tokenize(target)) // 2)
    )
    print(f"Reconstruction BLEU: {compute_bleu(rec, target):.2f}")
    print("Reconstruction:", rec)

    # tuned recon (no retraining)
    best_scale, rec_text, rec_bleu = extractor.tune_scale_for_reconstruction(z, target)
    print("Reconstruction (tuned):", rec_text)
    print(f"Tuned BLEU: {rec_bleu:.2f}")

    # transfer with tuned scale; try mild sampling to escape local modes
    result = extractor.steer_transfer(
        z,
        prompt='',
        max_new_tokens=50,
        steer_scale=best_scale,
        min_decode_len=5,
        temperature=0.9,
        top_k=5
    )
    print(f"Transfer result (tuned scale={best_scale}): {result}")

    # Optional batch (unchanged)
    try:
        df = pd.read_parquet("sample.parquet")
        if 'SMILES' not in df.columns:
            raise ValueError("The input file must contain a 'SMILES' column.")
        vectors, bleuscores, smiles_list = [], [], []
        for smiles in df['SMILES']:
            for seed in range(8):
                torch.manual_seed(seed)
                np.random.seed(seed)
                print(f"Processing SMILES: {smiles}")
                z = extractor.extract_zsteer(smiles, steps=500, lr=0.01)
                rec = extractor.steer_generate(
                    z,
                    max_new_tokens=len(smiles),
                    min_decode_len=max(0, len(smiles)//2)
                )
                bleu = compute_bleu(rec, smiles)
                print(f"BLEU (single-layer, no-embed, lr=1.0): {bleu:.2f}")
                bleuscores.append(bleu)
                vectors.append(z.cpu().detach().numpy())
                smiles_list.append(smiles)
        newdf = pd.DataFrame({'SMILES': smiles_list, 'steering_vector': vectors, 'bleu_score': bleuscores})
        outp = 'plot_sample_vectors.parquet'
        newdf.to_parquet(outp, index=False)
        print(f"Updated parquet saved to {outp}")
    except Exception as e:
        print(f"[batch] skipped: {e}")
