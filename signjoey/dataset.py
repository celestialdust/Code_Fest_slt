# coding: utf-8
"""
Data module
"""
from torchtext import data
from torchtext.data import Field, RawField
from typing import List, Tuple
import pickle
import gzip
import torch


def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object


class SignTranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
        self,
        path: str,
        fields: Tuple[RawField, RawField, Field, Field, Field],
        feature_type,
        **kwargs
    ):
        """Create a SignTranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [
                ("sequence", fields[0]),
                ("signer", fields[1]),
                ("sgn", fields[2]),
                ("gls", fields[3]),
                ("txt", fields[4]),
            ]

        if not isinstance(path, list):
            path = [path]

        samples = {}
        for annotation_file in path:
            tmp = load_dataset_file(annotation_file)
            for s in tmp:
                seq_id = s["name"]
                if seq_id in samples:
                    assert samples[seq_id]["name"] == s["name"]
                    assert samples[seq_id]["signer"] == s["signer"]
                    assert samples[seq_id]["gloss"] == s["gloss"]
                    assert samples[seq_id]["text"] == s["text"]
                    samples[seq_id]["sign"] = torch.cat(
                        [samples[seq_id]["sign"], s["sign"]], axis=1
                    )
                else:
                    if type(feature_type) == list:
                        kps = s["sign"]
                        kps[:, ::2] = 2 * ((kps[:, ::2] / 1920.0) - 0.5)
                        kps[:, 1::2] = 2 * ((kps[:, 1::2] / 1080.0) - 0.5)

                        final_kps = []
                        for ft in feature_type:
                            if ft == "body":
                                body = kps[:, 0:14 * 2]
                                final_kps = final_kps + [torch.from_numpy(body)]
                            elif ft == "face":
                                face = kps[:, 28:28 + 68 * 2]
                                final_kps = final_kps + [torch.from_numpy(face)]
                            elif ft == "hands":
                                hands = kps[:, 164:]
                                final_kps = final_kps + [torch.from_numpy(hands)]

                        sign_feature = torch.cat(final_kps, dim=1)


                    elif type(feature_type) == str and feature_type == "RGB":
                        sign_feature = s["sign"]


                    samples[seq_id] = {
                        "name": s["name"],
                        "signer": s["signer"],
                        "gloss": s["gloss"],
                        "text": s["text"],
                        "sign": sign_feature,
                    }

        examples = []
        for s in samples:
            sample = samples[s]
            examples.append(
                data.Example.fromlist(
                    [
                        sample["name"],
                        sample["signer"],
                        # This is for numerical stability
                        sample["sign"] + 1e-8,
                        str(sample["gloss"]).strip(),
                        str(sample["text"]).strip(),
                    ],
                    fields,
                )
            )
        super().__init__(examples, fields, **kwargs)
