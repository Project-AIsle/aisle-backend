from __future__ import annotations
from ..config import settings
 
import os
import math
from typing import List, Tuple, Optional, Dict, Any
import cv2
import onnxruntime as ort 
import numpy as np
import clip
import torch
from PIL import Image
import time

class Detector:
    def __init__(self, clip_classifier: CLIPClassifier, yolo_detector: YOLOXDetector, classes: List[str]):
        self._clip = clip_classifier
        self._yolo = yolo_detector
        self._classes = classes
    
    def infer(self, img_bytes: bytes):
        image_bgr = imdecode_image(img_bytes)
        boxes, scores = self._yolo.infer(image_bgr)
        if len(boxes) == 0:
            return ""
        prediction = ""
        for box in boxes:
            prediction = self._clip.classify_crop(image_bgr, box, self._classes)
        
        try:
            ann_img = draw_detections(image_bgr, boxes, scores, self._yolo.score_thr)
            ann_path = os.path.join(settings.DEBUG_OUT_DIR, f"frame_annotated_{int(time.time())*1000}.jpg")
            cv2.imwrite(ann_path, ann_img)
        except:
            pass

        return prediction

class CLIPClassifier:
    def __init__(self, model_name: str = , device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        # Carrega o modelo CLIP
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=self.device)
 
    def classify_crop(self, img_bgr: np.ndarray, box: np.ndarray, class_names: list) -> str:
        """
        Classifica o crop da imagem usando o modelo CLIP.
 
        Parameters:
        - img_bgr: Imagem em formato BGR.
        - box: Caixa de detecção [x1, y1, x2, y2].
        - class_names: Lista com as descrições de classes (ex. ['latinha de Red Bull']).
 
        Returns:
        - A classe predita pela comparação de CLIP.
        """
        # Converte BGR para RGB (necessário para CLIP)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
 
        # Faz o crop baseado na caixa de detecção
        x1, y1, x2, y2 = map(int, box)
        crop = img_rgb[y1:y2, x1:x2]
        crop_pil = Image.fromarray(crop)
 
        # Pré-processa o crop para CLIP
        crop_input = self.preprocess(crop_pil).unsqueeze(0).to(self.device)
 
        # Processa as classes de texto
        text_inputs = clip.tokenize(class_names).to(self.device)
 
        # Realiza a inferência
        with torch.no_grad():
            image_features = self.model.encode_image(crop_input)
            text_features = self.model.encode_text(text_inputs)
 
            # Calcula a similaridade entre imagem e texto
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).squeeze(0)
 
        # Obtém o índice da classe mais próxima
        best_class_idx = similarity.argmax().item()
        return class_names[best_class_idx]
 
class YOLOXDetector:
    """Wrapper para sessão ONNXRuntime do modelo YOLOX exportado no formato:
    - boxes: (1, N, 4) em coordenadas xyxy já no espaço do input (letterboxed)
    - scores: (1, N)
    - class_idx: (1, N) (opcional, não utilizado)
    """
 
    def __init__(self, model_path: str, providers: List[str], input_size: Tuple[int, int], score_thr: float, iou_thr: float) -> None:
        self.model_path = model_path
        self.providers = providers
        self.input_w, self.input_h = input_size
        self.score_thr = score_thr
        self.iou_thr = iou_thr
 
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Garante >= 1 thread mesmo se cpu_count() for None/0
        so.intra_op_num_threads = max(1, os.cpu_count() or 1)
        so.inter_op_num_threads = 1
 
        self.sess = ort.InferenceSession(self.model_path, sess_options=so, providers=self.providers)
        self.inp = self.sess.get_inputs()[0]
        self.in_name = self.inp.name
        self.outs = self.sess.get_outputs()
        self.out_names = [o.name for o in self.outs]
 
    # ---------- Pré/Pós ----------
    def _preprocess(self, img_bgr):
        img_pad, r, (padw, padh) = letterbox_bgr(img_bgr, (self.input_h, self.input_w))
        img = (img_pad[..., ::-1].astype(np.float32) / 255.0)  # BGR->RGB e normalização
        x = np.transpose(img, (2, 0, 1))[None, ...]  # NCHW
        h0, w0 = img_bgr.shape[:2]
        return x, (w0, h0, r, (padw, padh))
 
    def _postprocess(self, boxes: np.ndarray, scores: np.ndarray, meta):
        w0, h0, r, (padw, padh) = meta
        m = scores >= self.score_thr
        boxes, scores = boxes[m], scores[m]
        if boxes.size == 0:
            return boxes, scores
        keep = nms_xyxy(boxes, scores, self.iou_thr)
        boxes, scores = boxes[keep], scores[keep]
        # desfaz letterbox (tirar padding + dividir por escala)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - padw) / r
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - padh) / r
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w0 - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h0 - 1)
        return boxes, scores
 
    # ---------- Infer ----------
    def infer(self, img_bgr):
        x, meta = self._preprocess(img_bgr)
        out_list = self.sess.run(self.out_names, {self.in_name: x})
        out_map = {name: arr for name, arr in zip(self.out_names, out_list)}
 
        boxes = out_map.get("boxes")
        if boxes is None:
            boxes = out_list[0]
        scores = out_map.get("scores")
        if scores is None:
            scores = out_list[1]
 
        boxes = boxes[0]
        scores = scores[0]
        return self._postprocess(boxes, scores, meta)
 
def draw_detections(img_bgr: np.ndarray, boxes: np.ndarray, scores: np.ndarray, score_thr: float) -> np.ndarray:
    out = img_bgr.copy()
    for (x1, y1, x2, y2), sc in zip(boxes, scores):
        if sc < score_thr:
            continue
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{sc:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out
 
def hist_feature_hsv(img_bgr: np.ndarray, box_xyxy) -> np.ndarray:
    """Retorna histograma HSV normalizado (classe-agnóstico) para a box."""
    h0, w0 = img_bgr.shape[:2]
    x1, y1, x2, y2 = clip_box_xyxy(box_xyxy, w0, h0)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1,), dtype=np.float32)
    crop = img_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])
    feat = np.concatenate([h_hist.flatten(), s_hist.flatten(), v_hist.flatten()]).astype(np.float32)
    n = np.linalg.norm(feat) + 1e-9
    return feat / n
 
 
def iou_matrix(ref_boxes: np.ndarray, cur_boxes: np.ndarray) -> np.ndarray:
    if ref_boxes.size == 0 or cur_boxes.size == 0:
        return np.zeros((len(ref_boxes), len(cur_boxes)), dtype=np.float32)
    R, C = len(ref_boxes), len(cur_boxes)
    ious = np.zeros((R, C), dtype=np.float32)
    for i, (x1, y1, x2, y2) in enumerate(ref_boxes):
        area_i = (x2 - x1 + 1) * (y2 - y1 + 1)
        for j, (xx1, yy1, xx2, yy2) in enumerate(cur_boxes):
            ix1 = max(x1, xx1); iy1 = max(y1, yy1)
            ix2 = min(x2, xx2); iy2 = min(y2, yy2)
            w = max(0, ix2 - ix1 + 1); h = max(0, iy2 - iy1 + 1)
            inter = w * h
            area_j = (xx2 - xx1 + 1) * (yy2 - yy1 + 1)
            union = area_i + area_j - inter + 1e-9
            ious[i, j] = inter / union
    return ious
 
 
def match_by_iou_hist(
    img_ref: np.ndarray,
    ref_boxes: np.ndarray,
    ref_feats: List[np.ndarray],
    img_cur: np.ndarray,
    cur_boxes: np.ndarray,
    cur_feats: List[np.ndarray],
    iou_thr: float,
    hist_thr: float,
):
    """Para cada ref_box, acha cur_box com IoU >= iou_thr e correlação de hist >= hist_thr.
    Retorna (matched_indices, missing_ref_indices).
    """
    R, C = len(ref_boxes), len(cur_boxes)
    if R == 0:
        return np.array([], dtype=int), []
    if C == 0:
        return np.full(R, -1, dtype=int), list(range(R))
 
    ious = iou_matrix(ref_boxes, cur_boxes)
    matched = np.full(R, -1, dtype=int)
    used_cur = set()
 
    for i in range(R):
        cand = [j for j in range(C) if ious[i, j] >= iou_thr and j not in used_cur]
        if not cand:
            continue
        best_j, best_corr = -1, -2.0
        for j in cand:
            corr = float(cv2.compareHist(ref_feats[i], cur_feats[j], cv2.HISTCMP_CORREL))
            if corr > best_corr:
                best_corr, best_j = corr, j
        if best_j != -1 and best_corr >= hist_thr:
            matched[i] = best_j
            used_cur.add(best_j)
 
    missing = [i for i in range(R) if matched[i] == -1]
    return matched, missing 
 
def hist_feature_hsv(img_bgr: np.ndarray, box_xyxy) -> np.ndarray:
    """Retorna histograma HSV normalizado (classe-agnóstico) para a box."""
    h0, w0 = img_bgr.shape[:2]
    x1, y1, x2, y2 = clip_box_xyxy(box_xyxy, w0, h0)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1,), dtype=np.float32)
    crop = img_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])
    feat = np.concatenate([h_hist.flatten(), s_hist.flatten(), v_hist.flatten()]).astype(np.float32)
    n = np.linalg.norm(feat) + 1e-9
    return feat / n
 
 
def iou_matrix(ref_boxes: np.ndarray, cur_boxes: np.ndarray) -> np.ndarray:
    if ref_boxes.size == 0 or cur_boxes.size == 0:
        return np.zeros((len(ref_boxes), len(cur_boxes)), dtype=np.float32)
    R, C = len(ref_boxes), len(cur_boxes)
    ious = np.zeros((R, C), dtype=np.float32)
    for i, (x1, y1, x2, y2) in enumerate(ref_boxes):
        area_i = (x2 - x1 + 1) * (y2 - y1 + 1)
        for j, (xx1, yy1, xx2, yy2) in enumerate(cur_boxes):
            ix1 = max(x1, xx1); iy1 = max(y1, yy1)
            ix2 = min(x2, xx2); iy2 = min(y2, yy2)
            w = max(0, ix2 - ix1 + 1); h = max(0, iy2 - iy1 + 1)
            inter = w * h
            area_j = (xx2 - xx1 + 1) * (yy2 - yy1 + 1)
            union = area_i + area_j - inter + 1e-9
            ious[i, j] = inter / union
    return ious
 
 
def match_by_iou_hist(
    img_ref: np.ndarray,
    ref_boxes: np.ndarray,
    ref_feats: List[np.ndarray],
    img_cur: np.ndarray,
    cur_boxes: np.ndarray,
    cur_feats: List[np.ndarray],
    iou_thr: float,
    hist_thr: float,
):
    """Para cada ref_box, acha cur_box com IoU >= iou_thr e correlação de hist >= hist_thr.
    Retorna (matched_indices, missing_ref_indices).
    """
    R, C = len(ref_boxes), len(cur_boxes)
    if R == 0:
        return np.array([], dtype=int), []
    if C == 0:
        return np.full(R, -1, dtype=int), list(range(R))
 
    ious = iou_matrix(ref_boxes, cur_boxes)
    matched = np.full(R, -1, dtype=int)
    used_cur = set()
 
    for i in range(R):
        cand = [j for j in range(C) if ious[i, j] >= iou_thr and j not in used_cur]
        if not cand:
            continue
        best_j, best_corr = -1, -2.0
        for j in cand:
            corr = float(cv2.compareHist(ref_feats[i], cur_feats[j], cv2.HISTCMP_CORREL))
            if corr > best_corr:
                best_corr, best_j = corr, j
        if best_j != -1 and best_corr >= hist_thr:
            matched[i] = best_j
            used_cur.add(best_j)
 
    missing = [i for i in range(R) if matched[i] == -1]
    return matched, missing
 
def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int64)
 
def letterbox_bgr(img_bgr: np.ndarray, new_shape: Tuple[int, int] = (640, 640), color=(114, 114, 114)):
    h0, w0 = img_bgr.shape[:2]
    r = min(new_shape[0] / h0, new_shape[1] / w0)
    nh, nw = int(round(h0 * r)), int(round(w0 * r))
    im = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_shape[0] - nh) // 2
    bottom = new_shape[0] - nh - top
    left = (new_shape[1] - nw) // 2
    right = new_shape[1] - nw - left
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)
 
 
def clip_box_xyxy(box, w, h):
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(w - 1, x1)))
    x2 = int(max(0, min(w - 1, x2)))
    y1 = int(max(0, min(h - 1, y1)))
    y2 = int(max(0, min(h - 1, y2)))
    return x1, y1, x2, y2
 
 
def imdecode_image(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img