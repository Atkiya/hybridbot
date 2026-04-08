from rasa.engine.graph import GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.features import Features
from rasa.engine.graph import ExecutionContext
from rasa.shared.nlu.constants import TEXT
from typing import Any, Dict, List, Text, Optional
import numpy as np
import re
import logging

# pip install sentence-transformers
from sentence_transformers import SentenceTransformer

# bangla script detector
try:
    from bnlp import LanguageDetector as BnlpDetector
    _bnlp_detector = BnlpDetector()
    BNLP_READY = True
except Exception as e:
    logging.warning(f"bnlp_toolkit not available: {e}. Falling back to regex.")
    BNLP_READY = False


# Anchor phrases for language detection via embedding cosine similarity
_BANGLISH_ANCHORS = [
    "ami tomake bhalobashi",
    "apni kemon achen",
    "tumi ki korcho",
    "amar ektu problem hoyeche",
    "daam koto",
    "amar help lagbe",
    "ki hoyeche",
    "thik ache bhai",
    "ami achi",
    "ki hobe",
    "bolo",
    "ki bolcho",
    "kemon acho",
    "ami bujhte parchi na",
    "sahajjo koro",
    "admission er khabar dao",
    "fees koto",
    "course dekhao",
    "program ache",
    "scholarship pabo",
    "result kobe",
    "class kobe",
    "exam schedule dao",
    "notun student",
    "valo lagche",
    "dhonnobad",
    "shukriya",
    "help chai",
    "jana dorkar",
]

_ENGLISH_ANCHORS = [
    "how can I help you today",
    "what is the price of this product",
    "I need some assistance please",
    "can you show me the details",
    "I would like to place an order",
    "please help me with this issue",
    "what are the available options",
    "I want to check my account",
    "show me the schedule",
    "what are the admission requirements",
    "tell me about the programs",
    "how much is the tuition fee",
]

_BANGLA_ANCHORS = [
    "আমি আপনাকে ভালোবাসি",
    "আপনি কেমন আছেন",
    "আমার একটু সমস্যা হয়েছে",
    "দাম কত",
    "আমার সাহায্য দরকার",
    "কী হয়েছে",
    "ঠিক আছে ভাই",
    "আমি বুঝতে পারছি না",
    "ভর্তির তথ্য দিন",
    "কোর্স দেখান",
    "পরীক্ষার সময়সূচী",
    "বৃত্তি পাবো",
    "ক্লাস কখন",
    "ফলাফল কবে",
    "ধন্যবাদ",
]


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER,
    is_trainable=False,
)
class MultilingualFeaturizer(GraphComponent):
    MODEL_NAME = "intfloat/multilingual-e5-small"
    QUERY_PREFIX = "query: "

    def __init__(self):
        logging.info(f"[MultilingualFeaturizer] Loading {self.MODEL_NAME} …")
        self._embedder = SentenceTransformer(self.MODEL_NAME)
        logging.info("[MultilingualFeaturizer] Model ready.")

      
        self._banglish_centroid = self._l2_norm(
            self._embed(_BANGLISH_ANCHORS).mean(axis=0)
        )
        self._english_centroid = self._l2_norm(
            self._embed(_ENGLISH_ANCHORS).mean(axis=0)
        )
        self._bangla_centroid = self._l2_norm(
            self._embed(_BANGLA_ANCHORS).mean(axis=0)
        )

   

    @staticmethod
    def create(
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> "MultilingualFeaturizer":
        return MultilingualFeaturizer()

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        self.process(training_data.training_examples)
        return training_data

    
    # Embedding helpers
    

    def _embed(self, texts: List[str]) -> np.ndarray:
        prefixed = [self.QUERY_PREFIX + t for t in texts]
        return self._embedder.encode(
            prefixed,
            normalize_embeddings=True, 
            batch_size=32,
            show_progress_bar=False,
        )

    @staticmethod
    def _l2_norm(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-9 else vec

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    
    # Language detection
    

    def _has_bangla_script(self, text: str) -> bool:
        return bool(re.search(r'[\u0980-\u09FF]', text))

    def is_bangla(self, text: str, text_vec: np.ndarray) -> bool:
        if BNLP_READY:
            try:
                return _bnlp_detector.is_bengali(text)
            except Exception:
                pass

        if self._has_bangla_script(text):
            return True

        # Embedding fallback
        sim_bn = self._cosine(text_vec, self._bangla_centroid)
        sim_bl = self._cosine(text_vec, self._banglish_centroid)
        sim_en = self._cosine(text_vec, self._english_centroid)
        return sim_bn > sim_bl + 0.05 and sim_bn > sim_en + 0.05

    def is_banglish(self, text: str, text_vec: np.ndarray) -> bool:
        if self._has_bangla_script(text):
            return False  

      
        if len(text.strip()) < 2:
            return False  

       
        MARGIN = 0.04
        sim_banglish = self._cosine(text_vec, self._banglish_centroid)
        sim_english  = self._cosine(text_vec, self._english_centroid)

        logging.debug(
            f"[MultilingualFeaturizer] banglish_sim={sim_banglish:.3f}  "
            f"english_sim={sim_english:.3f}  text={text!r}"
        )
        return sim_banglish > sim_english + MARGIN

    def _detect_language(self, text: str, text_vec: np.ndarray) -> str:
        """Return 'bn' | 'banglish' | 'en'."""
        if self.is_bangla(text, text_vec):
            return "bn"
        if self.is_banglish(text, text_vec):
            return "banglish"
        return "en"

    
    # Dense feature injection
    

    def _set_dense_features(self, message: Message, embedding: np.ndarray) -> None:
        sentence_vec = embedding.reshape(1, -1).astype(np.float32)  # (1, 384)
        feature = Features(
            features=sentence_vec,
            feature_type="sentence",
            attribute=TEXT,
            origin=self.__class__.__name__,
        )
        message.add_features(feature)

    
    # Entity injection
    

    def _inject_language_entity(
        self, message: Message, lang_value: str, original_text: str
    ) -> None:
        existing = message.get("entities") or []
        # always overwrite so the slot stays current each turn
        existing = [e for e in existing if e.get("entity") != "user_language"]
        existing.append({
            "entity":     "user_language",
            "value":      lang_value,
            "confidence": 1.0,
            "extractor":  "MultilingualFeaturizer",
            "start":      0,
            "end":        len(original_text),
        })
        message.set("entities", existing)
        logging.info(f"[MultilingualFeaturizer] Detected language='{lang_value}' for: {original_text!r}")

    
    # Main pipeline entry point
    

    def process(self, messages: List[Message]) -> List[Message]:
        texts = [m.get("text") or "" for m in messages]

        non_empty_indices = [i for i, t in enumerate(texts) if t]
        embeddings: Dict[int, np.ndarray] = {}

        if non_empty_indices:
            batch_texts = [texts[i] for i in non_empty_indices]
            batch_vecs  = self._embed(batch_texts)  
            embeddings  = dict(zip(non_empty_indices, batch_vecs))

        for idx, message in enumerate(messages):
            text = texts[idx]
            if not text:
                continue

            embedding: np.ndarray = embeddings[idx]  

            # detect language 
            lang = self._detect_language(text, embedding)

            # attach dense features for DIET
            self._set_dense_features(message, embedding)

            # inject language entity for slot tracking 
            self._inject_language_entity(message, lang, text)

        return messages
