import h5py
import pytest 
import numpy
import logging

logger = logging.getLogger(__name__)

def compare_dataset(question_id, file_id, dataset_id, sample_output_dir, student_output_dir):
    """Test whether dataset values in reference and student outputs match."""

    reference_file = sample_output_dir / question_id / (file_id + ".hdf5")
    test_file = student_output_dir / question_id / (file_id + ".hdf5")
    
    assert reference_file.exists() and reference_file.is_file()
    assert test_file.exists() and test_file.is_file() 

    with h5py.File(reference_file, "r") as ref, h5py.File(test_file, "r") as test:
        ref_data = ref[dataset_id][()]
        test_data = test[dataset_id][()]

        logger.debug("\n" + str(ref_data))
        logger.debug("\n" + str(test_data))  
        
        assert ref_data.shape == test_data.shape, (
            f"Shape of reference data: {ref_data.shape} does not match shape of test data {test_data.shape}"
        )
        numpy.testing.assert_allclose(test_data, ref_data, rtol=1e-4, atol=1e-2) 
      
@pytest.mark.parametrize("dataset_id", [
    "S", 
    "gamma", 
    "H_core",
    "Fa_initial",
    "Fb_initial",
    "Ea",
    "Eb", 
    "electronic_energy", 
    "nuclear_energy", 
    "total_energy",
    ])
@pytest.mark.parametrize("file_id", ["H2","HF","HO"])
@pytest.mark.parametrize("question_id", ["hw_5_1"])
def test_hw_3_1(question_id, file_id, dataset_id, sample_output_dir, student_output_dir):
    compare_dataset(question_id, file_id, dataset_id, sample_output_dir, student_output_dir)