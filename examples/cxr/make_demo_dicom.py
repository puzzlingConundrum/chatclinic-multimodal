from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid


HERE = Path(__file__).resolve().parent
SOURCE_PNG = HERE / "torchxrayvision_00000001_000.png"
OUTPUT_DCM = HERE / "torchxrayvision_00000001_000_demo.dcm"


def main() -> None:
    image = Image.open(SOURCE_PNG).convert("L")
    pixels = np.asarray(image, dtype=np.uint8)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    dataset = FileDataset(str(OUTPUT_DCM), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.is_little_endian = True
    dataset.is_implicit_VR = False

    dataset.PatientName = "ChatClinic^Demo"
    dataset.PatientID = "CHATCLINIC-DEMO-CXR"
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "DX"
    dataset.StudyDate = "20260101"
    dataset.StudyTime = "000000"
    dataset.SeriesNumber = "1"
    dataset.InstanceNumber = "1"
    dataset.StudyDescription = "Derived demo chest radiograph"
    dataset.SeriesDescription = "TorchXRayVision PNG converted to demo DICOM"

    dataset.Rows, dataset.Columns = pixels.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.PixelData = pixels.tobytes()

    dataset.save_as(str(OUTPUT_DCM), write_like_original=False)
    print(OUTPUT_DCM)


if __name__ == "__main__":
    main()
