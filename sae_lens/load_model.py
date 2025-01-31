from typing import Any, cast

import torch
from transformer_lens import HookedTransformer
try:
    from transformer_lens import HookedChameleon
except Exception as e:
    HookedChameleon = None
try:
    from transformer_lens.HookedLlava import HookedLlava
except Exception as e:
    HookedLlava = None
from transformer_lens.hook_points import HookedRootModule
from transformers import AutoModelForSeq2SeqLM,AutoModelForCausalLM

def load_model(
    model_class_name: str,
    model_name: str,
    device: str | torch.device | None = None,
    model_from_pretrained_kwargs: dict[str, Any] | None = None,
    local_model_path: str | None = None,
) -> HookedRootModule:
    model_from_pretrained_kwargs = model_from_pretrained_kwargs or {}

    if "n_devices" in model_from_pretrained_kwargs:
        n_devices = model_from_pretrained_kwargs["n_devices"]
        if n_devices > 1:
            print("MODEL LOADING:")
            print("Setting model device to cuda for d_devices")
            print(f"Will use cuda:0 to cuda:{n_devices-1}")
            device = "cuda"
            print("-------------")

    if local_model_path is not None:
        if model_class_name == "HookedChameleon":
            from transformers import ChameleonForConditionalGeneration
            hf_model = ChameleonForConditionalGeneration.from_pretrained(local_model_path)
        elif model_class_name =="HookedLlava" and "llava" in model_name:
            from transformers import LlavaForConditionalGeneration
            hf_model=LlavaForConditionalGeneration.from_pretrained(local_model_path)
        elif model_class_name =="HookedLlava" and "mistralai/Mistral-7B-Instruct-v0.2" in model_name:
            hf_model = AutoModelForCausalLM.from_pretrained(local_model_path)
        else:
            hf_model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path)
    else:
        hf_model = None
        

    if model_class_name == "HookedTransformer":
        return HookedTransformer.from_pretrained_no_processing(
            model_name=model_name, device=device, **model_from_pretrained_kwargs
        )
    elif model_class_name == "HookedMamba":
        try:
            from mamba_lens import HookedMamba
        except ImportError:  # pragma: no cover
            raise ValueError(
                "mamba-lens must be installed to work with mamba models. This can be added with `pip install sae-lens[mamba]`"
            )
        # HookedMamba has incorrect typing information, so we need to cast the type here
        return cast(
            HookedRootModule,
            HookedMamba.from_pretrained(
                model_name, device=cast(Any, device), **model_from_pretrained_kwargs
            ),
        )
    elif model_class_name == "HookedChameleon":
        if HookedChameleon is None:
            raise ValueError("HookedChameleon is not installed")
        if hf_model is None:
            return HookedChameleon.from_pretrained(
                model_name=model_name, device=device, **model_from_pretrained_kwargs
            )
        else:
            return HookedChameleon.from_pretrained(
                model_name=model_name, hf_model=hf_model, 
                device=device, **model_from_pretrained_kwargs
            )
    elif model_class_name == "HookedLlava" and "llava" in model_name:
        if HookedLlava is None:
            raise ValueError("HookedLlava is not installed")
        if hf_model is None:
            Warning("no hf_model for hookllava")
            return HookedLlava.from_pretrained(
                model_name=model_name, device=device, **model_from_pretrained_kwargs
            )
        else:
            vision_tower=hf_model.vision_tower
            multi_modal_projector = hf_model.multi_modal_projector
            model=HookedLlava.from_pretrained(
                model_name=model_name, hf_model=hf_model.language_model, 
                device=device,vision_tower = vision_tower,multi_modal_projector =multi_modal_projector, **model_from_pretrained_kwargs
            )
            del hf_model
            torch.cuda.empty_cache()
            print("clear hf model")
            return model
    elif model_class_name == "HookedLlava" and "mistralai/Mistral-7B-Instruct-v0.2" in model_name:
        if HookedLlava is None:
            raise ValueError("HookedLlava is not installed")
        if hf_model is None:
            Warning("no hf_model for hookllava")
            return HookedLlava.from_pretrained(
                model_name=model_name, device=device, **model_from_pretrained_kwargs
            )
        else:
            return HookedLlava.from_pretrained(
                model_name=model_name, hf_model=hf_model, fold_ln=False,
                center_writing_weights=False,center_unembed=False,
                device=device, **model_from_pretrained_kwargs)
    else:  # pragma: no cover
        raise ValueError(f"Unknown model class: {model_class_name}")
