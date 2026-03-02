from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import onnxruntime as ort

def pick_providers(prefer: str = "auto") -> List[str]:
    prefer = (prefer or "auto").lower()
    avail = ort.get_available_providers()
    if prefer == "cpu":
        return ["CPUExecutionProvider"]
    if prefer == "directml":
        return ["DmlExecutionProvider","CPUExecutionProvider"] if "DmlExecutionProvider" in avail else ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider","CPUExecutionProvider"]
    if "DmlExecutionProvider" in avail:
        return ["DmlExecutionProvider","CPUExecutionProvider"]
    return ["CPUExecutionProvider"]

@dataclass
class ORTModel:
    path: str
    prefer: str = "auto"
    input_names_override: Optional[List[str]] = None
    output_name_override: str = ""

    def __post_init__(self):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.providers = pick_providers(self.prefer)
        self.sess = ort.InferenceSession(self.path, sess_options=so, providers=self.providers)

        ins = self.sess.get_inputs()
        outs = self.sess.get_outputs()

        self.input_names = [i.name for i in ins]
        self.input_shapes = [i.shape for i in ins]
        self.output_names = [o.name for o in outs]

        if self.input_names_override:
            self.input_names = list(self.input_names_override)
        self.output_name = self.output_name_override or self.output_names[0]

    def run(self, feed: Dict[str, Any]):
        return self.sess.run([self.output_name], feed)[0]
