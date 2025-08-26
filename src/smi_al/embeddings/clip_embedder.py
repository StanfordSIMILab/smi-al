from typing import List
import numpy as np

# Try OpenCLIP; otherwise fall back
def _try_openclip():
    try:
        import torch, open_clip
        return torch, open_clip
    except Exception:
        return None, None

def embed_images(paths: List[str], model_name: str = 'ViT-B-32', pretrained: str = 'laion2b_s34b_b79k') -> np.ndarray:
    torch, open_clip = _try_openclip()
    if torch is not None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        model.eval()
        embs = []
        for p in paths:
            img = preprocess(open_clip.open_image(p)).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode_image(img)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            embs.append(feat.squeeze(0).detach().cpu().numpy())
        return np.stack(embs, axis=0).astype(np.float32)
    # Fallback: HOG + color histogram
    import cv2
    embs = []
    hog = cv2.HOGDescriptor()
    for p in paths:
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(p)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv],[0,1,2],None,[8,8,8],[0,180,0,256,0,256]).flatten()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h = hog.compute(gray)
        vec = np.concatenate([hist/ (hist.sum()+1e-6), h.flatten()])
        embs.append(vec.astype(np.float32))
    # L2 normalize
    X = np.stack(embs, axis=0)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return X
