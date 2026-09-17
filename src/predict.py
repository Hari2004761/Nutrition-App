"""Run the trained classifier on one image and print its top predictions.

    python src/predict.py path/to/food_photo.jpg
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_large

MODELS = Path("models")

eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(checkpoint_path, device):
    """Load a checkpoint and return (model, classes) ready for inference."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]

    model = mobilenet_v3_large(weights=None)
    model.classifier[3] = torch.nn.Linear(
        model.classifier[3].in_features, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    return model, classes


def predict_pil(model, classes, pil_image, device, top_k=3):
    """Run the classifier on an already-loaded PIL Image (no disk I/O)."""
    x = eval_tf(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]
    top_k = min(top_k, len(classes))
    top_probs, top_idx = probs.topk(top_k)
    return [(classes[i], float(p)) for p, i in zip(top_probs, top_idx)]


def predict(model, classes, image_path, device, top_k=3):
    img = Image.open(image_path).convert("RGB")
    return predict_pil(model, classes, img, device, top_k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="Path to a food photo")
    ap.add_argument("--checkpoint", default=str(MODELS / "classifier_best.pt"),
                    help="Path to a saved model checkpoint")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    if not Path(args.checkpoint).exists():
        raise SystemExit(f"No checkpoint at {args.checkpoint}. Train first, "
                          f"or point --checkpoint at models/capped_run/classifier_best.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, classes = load_model(args.checkpoint, device)

    results = predict(model, classes, args.image, device, args.top_k)

    print(f"\nImage: {args.image}")
    print(f"Model: {args.checkpoint}\n")
    for i, (label, prob) in enumerate(results):
        marker = "->" if i == 0 else "  "
        print(f"  {marker} {label:<20} {prob*100:5.1f}%")


if __name__ == "__main__":
    main()