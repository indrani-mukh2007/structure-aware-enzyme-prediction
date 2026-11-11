from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class DiamondHit:
    query_id: str
    subject_id: str
    pident: float
    alignment_length: int
    evalue: float
    bitscore: float


class DiamondHomologyBaseline:
    def __init__(self, num_classes: int, fallback_prior: Optional[np.ndarray] = None):
        self.num_classes = num_classes
        self.train_labels: Dict[str, np.ndarray] = {}
        self.fallback_prior = fallback_prior

    @staticmethod
    def get_diamond_commands(
        train_fasta: str,
        test_fasta: str,
        db_path: str = "train_swissprot_db.dmnd",
        out_tsv: str = "diamond_test_hits.tsv",
        evalue: float = 1e-3,
        max_target_seqs: int = 50
    ) -> Tuple[str, str]:
        makedb_cmd = f"diamond makedb --in {train_fasta} -d {db_path}"
        blastp_cmd = (
            f"diamond blastp -d {db_path} -q {test_fasta} -o {out_tsv} "
            f"--ultra-sensitive --evalue {evalue} --max-target-seqs {max_target_seqs} "
            f"--outfmt 6 qseqid sseqid pident length evalue bitscore"
        )
        return makedb_cmd, blastp_cmd

    def fit(self, train_ids: List[str], train_ec_matrix: np.ndarray) -> None:
        assert len(train_ids) == train_ec_matrix.shape[0]
        self.train_labels = {
            pid: train_ec_matrix[i].astype(np.float32)
            for i, pid in enumerate(train_ids)
        }
        self.fallback_prior = np.mean(train_ec_matrix, axis=0).astype(np.float32)

    @staticmethod
    def parse_diamond_outfmt6(tsv_filepath: str) -> Dict[str, List[DiamondHit]]:
        hits_by_query: Dict[str, List[DiamondHit]] = {}
        with open(tsv_filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 6:
                    continue
                hit = DiamondHit(
                    query_id=parts[0],
                    subject_id=parts[1],
                    pident=float(parts[2]),
                    alignment_length=int(parts[3]),
                    evalue=float(parts[4]),
                    bitscore=float(parts[5])
                )
                hits_by_query.setdefault(hit.query_id, []).append(hit)
        return hits_by_query

    def predict_from_hits(
        self,
        test_ids: List[str],
        hits_by_query: Dict[str, List[DiamondHit]],
        top_k: Optional[int] = None
    ) -> np.ndarray:
        preds = np.zeros((len(test_ids), self.num_classes), dtype=np.float32)

        for idx, qid in enumerate(test_ids):
            hits = hits_by_query.get(qid, [])
            valid_hits = [h for h in hits if h.subject_id in self.train_labels and h.bitscore > 0]

            if not valid_hits:
                if self.fallback_prior is not None:
                    preds[idx] = self.fallback_prior
                continue

            valid_hits.sort(key=lambda h: h.bitscore, reverse=True)
            if top_k is not None:
                valid_hits = valid_hits[:top_k]

            total_bitscore = sum(h.bitscore for h in valid_hits)
            weighted_sum = np.zeros(self.num_classes, dtype=np.float32)
            for h in valid_hits:
                weighted_sum += h.bitscore * self.train_labels[h.subject_id]

            preds[idx] = weighted_sum / max(total_bitscore, 1e-8)

        return preds
