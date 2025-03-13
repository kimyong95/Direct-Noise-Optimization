from .rewards_zoo import *

RFUNCTIONS = {
    "aesthetic" : aesthetic_loss_fn,
    "hps" : hps_loss_fn,
    "pick" : pick_loss_fn,
    "white" : white_loss_fn,
    "black" : black_loss_fn,
    "compressibility": jpeg_compressibility,
    "incompressibility": jpeg_incompressibility,
    "gemini-binary": gemini_binary,
    "llama-binary": llama_binary,
    "gemini": gemini,
}