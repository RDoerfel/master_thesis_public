from src import json_handling 
import pandas as pd
from pathlib import Path
from src.paths import Paths

LOCATION = Path(__file__).parent
VALIDATION_FILE = Path(LOCATION,'resources','FreeSurferColorLut.txt')

def read_rois(roi_file,validate=True):
    """ Read the roi file and return a dictionary with the roi names and the
    corresponding indices.
    args:
        roi_file (str): path to the roi file
        validate (bool): if True, validate the roi file (default: True)
    returns:
        rois (dict): dictionary with the roi names and the corresponding indices"""
    rois = json_handling.read_json(roi_file)
    if validate:
        validate_rois(rois)
    return rois

def read_roi_names(roi_file):
    """ Read the roi file and return a list with the roi names.
    args:
        roi_file (str): path to the roi file
    returns:
        roi_names (list): list with the roi names"""
    rois = read_rois(roi_file)
    roi_names = list(rois.keys())
    return roi_names

def save_to_roi_lut(rois,save_dir, name):
    """ Save as a look up table for the rois to be used in NRUs (Jonas') pipeline
    args:
        rois (dict): dictionary with the roi names and the corresponding indices
        save_dir (str): path to the directory where the roi look up table should be saved
        name (str): name of the roi selection
        save (bool): if True, save to excel sheet (default: True)"""
    fs_index = []
    pve_index = []
    pve_name = []

    roi_keys = list(rois.keys())

    for i,region in enumerate(roi_keys):
        hemis = list(rois[region].keys())
        for hemi in rois[region]['hemi']:
            n_labels = len(rois[region]['hemi'][hemi]['index'])
            fs_index = fs_index + rois[region]['hemi'][hemi]['index']
            pve_index = pve_index + n_labels*[i+1]
            pve_name = pve_name + n_labels*[region]

    df = pd.DataFrame({'FS_index':fs_index,'PVE_index':pve_index,'PVE_name':pve_name})
    file_name = f'ROI_LUT_FS_2_PVE_{name}.xlsx'
    file = Path(save_dir,file_name)
    df.to_excel(file,index=False)

def save_regions_to_excel(rois,save_dir, name, hemi):
    """ Same as save_to_roi_lut, but with label name instead of pve_name"""
    fs_index = []
    label_name = []
    roi_name = []

    roi_keys = list(rois.keys())
    for i,region in enumerate(roi_keys):
        hemis = list(rois[region]['hemi'].keys())   
        if hemi in hemis:
            n_labels = len(rois[region]['hemi'][hemi]['index'])
            fs_index = fs_index + rois[region]['hemi'][hemi]['index']
            label_name = label_name + rois[region]['hemi'][hemi]['label']
            roi_name = roi_name + n_labels*[region]


    

    df = pd.DataFrame({'ROI':roi_name,'Label':label_name,'Index':fs_index})
    file_name = f'roi_names_{name}.xlsx'
    file = Path(save_dir,file_name)
    df.to_excel(file,index=False)

def print_regions(rois):
    """ Print the regions in the roi file
    args:
        rois (dict): dictionary with the roi names and the corresponding indices"""
    roi_keys = list(rois.keys())
    print(f"Using {len(roi_keys)} regions:")
    for i,region in enumerate(roi_keys):
        print(f"{region} ({rois[region]['location']})")

def validate_rois(rois):
    """ Validate the roi file
    args:
        rois (dict): dictionary with the roi names and the corresponding indices"""
    
    dfFSRoiLut = pd.read_csv(VALIDATION_FILE,delim_whitespace=True, lineterminator='\n')
    dfFSRoiLut.set_index("#No.",inplace=True)

    for region in rois:
        for hemi in rois[region]['hemi']:
            indices = rois[region]['hemi'][hemi]['index']
            labels = rois[region]['hemi'][hemi]['label']
            assert len(indices) == len(labels), f"Number of indices and labels do not match. Indices: {len(indices)}, Labels: {len(labels)}."

            for i in range(len(indices)):
                label_expected = labels[i]
                index = indices[i]
                label_actual = dfFSRoiLut.loc[index]['Label']
                assert label_expected == label_actual, f"Labels do not match. Expected: {label_expected}, Actual: {label_actual}."
