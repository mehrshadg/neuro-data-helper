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
    "template.wang2015.59k.dlabel.nii",
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


names = [
    "topo.t1t2.59k.dscalar.nii",
    "topo.margulies2016.32k.dscalar.nii",
    "topo.margulies2016.59k.dscalar.nii",

]
for name in names:
    print(name)
    name = root + name
    left = name.replace(".dscalar.nii", "_cortexL.func.gii")
    right = name.replace(".dscalar.nii", "_cortexR.func.gii")
    merged = name.replace(".dscalar.nii", "_cortex.dscalar.nii")
    os.system(f"wb_command -cifti-separate {name} COLUMN -metric CORTEX_LEFT {left} -metric CORTEX_RIGHT {right}")
    os.system(f"wb_command -cifti-create-dense-scalar {merged} -left-metric {left} -right-metric {right}")
    os.system(f"rm {left} {right}")