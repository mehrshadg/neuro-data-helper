import os

root = "/Users/Mehrshad/PycharmProjects/help/neuro_helper/assets/hcp1200/"
names = [
    "template.cole.36012.32k.dlabel.nii",
    "template.cole.36012.4k.dlabel.nii",
    "template.cole.36012.59k.dlabel.nii",
    "template.schaefer2018.20017.32k.dlabel.nii",
    "template.schaefer2018.2007.32k.dlabel.nii",
    "template.schaefer2018.2007.59k.dlabel.nii",
    "template.wang2015.32k.dlabel.nii",
    "template.wang2015.59k.dlabel.nii"
]
for name in names:
    print(name)
    name = root + name
    left = name.replace(".dlabel.nii", "_cortexL.label.gii")
    right = name.replace(".dlabel.nii", "_cortexR.label.gii")
    merged = name.replace(".dlabel.nii", "_cortex.dlabel.nii")
    os.system(f"wb_command -cifti-separate {name} COLUMN -label CORTEX_LEFT {left} -label CORTEX_RIGHT {right}")
    os.system(f"wb_command -cifti-create-label {merged} -left-label {left} -right-label {right}")
    os.system(f"rm {left} {right}")