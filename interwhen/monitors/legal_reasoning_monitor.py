import asyncio
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from .base import VerifyMonitor


# Your modules
from interwhen.legal.state_extractor import extract_state
from interwhen.legal.verifier import verify_state

logger = logging.getLogger(__name__)

MODEL_NAME = "DChak2000/minilm-vu-classifier"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MAX_LEN = 512


class LegalReasoningMonitor(VerifyMonitor):

    def __init__(
        self,
        name: str,
        example_id: str,
        threshold: float = 0.7,
    ):
        super().__init__(name)

        self.example_id = example_id
        self.threshold = threshold

        logger.info("Loading MiniLM VU detector...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(MODEL_NAME)
            .to(DEVICE)
        )

        self.model.eval()

        logger.info("MiniLM loaded.")

        self.current_line = ""
        self.last_checkpoint = ""

        #
        # Allow only one verifier at a time.
        #

        self.verification_in_progress = False

        self.lock = asyncio.Lock()

    # MiniLM Prediction
    @torch.no_grad()
    def predict(self, reasoning: str):

        inputs = self.tokenizer(
            reasoning,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(DEVICE)
            for k, v in inputs.items()
        }

        logits = self.model(**inputs).logits

        probs = torch.softmax(
            logits,
            dim=-1,
        )[0]

        vu_probability = probs[1].item()

        return {
            "checkpoint":
                vu_probability >= self.threshold,
            "probability":
                vu_probability,
        }

    # Step Extractor
    def step_extractor(
        self,
        chunk: str,
        generated_text: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Every completed line is a candidate checkpoint.

        The accumulated reasoning is passed through MiniLM.

        If MiniLM predicts VU=1,
        Interwhen will invoke verify().
        """

        self.current_line += chunk
        #
        # Ignore new checkpoints while a verifier is running.
        #

        if self.verification_in_progress:
            return False, None

        while "\n" in self.current_line:

            idx = self.current_line.index("\n")

            #
            # Remaining unfinished text
            #

            self.current_line = self.current_line[idx + 1:]

            checkpoint = generated_text[
                : len(generated_text) - len(self.current_line)
            ].rstrip()

            if not checkpoint.strip():
                continue

            #
            # Don't verify the final prediction.
            #

            last = (
                checkpoint
                .splitlines()[-1]
                .strip()
                .lower()
            )

            if last in (
                "entailment",
                "contradiction",
            ):
                continue

            prediction = self.predict(
                checkpoint
            )

            logger.info("=" * 80)
            logger.info(
                "[Legal] Candidate checkpoint "
                "(VU probability %.3f)",
                prediction["probability"],
            )
            logger.info(
                "[Legal] Current reasoning prefix:\n%s",
                checkpoint,
            )
            logger.info("=" * 80)

            if prediction["checkpoint"]:

                self.last_checkpoint = checkpoint

                logger.info(
                    "[Legal] Accepted checkpoint."
                )

                return True, checkpoint

        return False, None
    


    # Verification
    async def verify(
        self,
        reasoning: str,
        token_index: int,
        event: asyncio.Event,
        event_info: dict,
    ):

        async with self.lock:

            if event.is_set():
                return

            #
            # Another verifier is already active.
            #

            if self.verification_in_progress:
                return

            self.verification_in_progress = True

        try:

            logger.info("=" * 80)
            logger.info(
                "[%s] Verifying checkpoint",
                self.example_id,
            )
            logger.info("=" * 80)

            logger.info(reasoning)

            # ----------------------------------------------------
            # State extraction
            # ----------------------------------------------------

            try:

                logger.info("=" * 80)
                logger.info("Calling state extractor...")
                logger.info("Reasoning sent to extractor:")
                logger.info(reasoning)
                logger.info("=" * 80)

                vu = await asyncio.to_thread(
                    extract_state,
                    reasoning,
                )

            except Exception as e:

                logger.exception(
                    "State extraction failed: %s",
                    e,
                )

                return

            if vu is None:

                logger.info(
                    "[Legal] No complete VU found."
                )

                return

            logger.info(
                "Extracted VU:\n%s",
                vu,
            )

            # ----------------------------------------------------
            # Verification
            # ----------------------------------------------------

            try:

                result = await asyncio.to_thread(
                    verify_state,
                    self.example_id,
                    vu,
                )

            except Exception as e:

                logger.exception(
                    "Verifier crashed: %s",
                    e,
                )

                return

            status = result.get(
                "status",
                "FAIL",
            )

            logger.info(
                "Verifier returned %s",
                status,
            )

            #
            # PASS
            #

            if status == "PASS":
                return

            #
            # FAIL
            #

            feedback = result.get(
                "feedback",
                None,
            )

            if feedback is None:

                feedback = """
    The previous reasoning contains a legal reasoning step
    that is inconsistent with the statute.

    Reconsider your most recent inference.

    Do not repeat the same reasoning.

    Continue from the corrected reasoning.
    """.strip()

            feedback = (
                "\n\n"
                "[LEGAL VERIFIER]\n"
                f"{feedback}\n"
                "[/LEGAL VERIFIER]\n\n"
            )

            if not event.is_set():

                event_info["generated_text"] = reasoning
                event_info["feedback"] = feedback
                event_info["correction_index"] = token_index

                event.set()

            return

        finally:

            #
            # Always allow the next verifier to run,
            # even if an exception or early return occurred.
            #

            async with self.lock:
                self.verification_in_progress = False

    
    # Fix
    

    async def fix(
        self,
        generated_text: str,
        event_info: dict,
        fix_method=None,
    ):

        feedback = event_info.get(
            "feedback",
            None,
        )

        generation = event_info.get(
            "generated_text",
            generated_text,
        )

        if feedback is None:

            return generation

        logger.info(
            "[Legal] Injecting verifier feedback."
        )

        return generation + feedback