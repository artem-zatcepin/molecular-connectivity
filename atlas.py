from pathlib import Path
import numpy as np
import pandas as pd

import nibabel as nib


class Atlas:

    def __init__(self,
                 atlas_dir,
                 atlas_name=None,):
        atlas_dir = Path(atlas_dir)
        self.dir = atlas_dir
        if atlas_name is None:
            self.name = atlas_dir.name
        else:
            self.name = atlas_name

        # Read atlas
        if atlas_dir.suffix == '.nii' or atlas_dir.suffix == '.gz':
            self.path = atlas_dir
            self.nim = nib.squeeze_image(nib.load(self.path))
        else:
            try:
                self.path = atlas_dir / f'{self.name}.nii'
                self.nim = nib.squeeze_image(nib.load(self.path))
            except FileNotFoundError:
                self.path = atlas_dir / f'{self.name}.nii.gz'
                self.nim = nib.squeeze_image(nib.load(self.path))
        self.arr = self.nim.get_fdata()
        self.n_dim = self.nim.ndim

        # Read legend
        try:
            # try excel first
            self.legend_path = atlas_dir / f'{self.name}.xlsx'
            legend = pd.read_excel(self.legend_path)
            legend = legend.loc[legend['Analysis'] == 1]
            legend.sort_values('Order', inplace=True)
            self.legend = legend.set_index('BrainRegion').to_dict()['Label']
            self.vois = list(self.legend.keys())
            network_names = [col for col in legend if col.startswith('Network:')]
            if len(network_names) > 0:
                self.networks = {}
                for network_name in network_names:
                    network_regions = legend.loc[legend[network_name] == 1]['BrainRegion'].to_list()
                    self.networks[network_name.split()[1]] = network_regions
        except (FileNotFoundError, NotADirectoryError, pd.errors.ParserError):
            try:
                self.legend_path = atlas_dir / f'{self.name}.txt'
                legend = pd.read_csv(self.legend_path, sep='\t', header=None)
                self.legend = legend.set_index(0).to_dict()[2]
                self.vois = list(self.legend.keys())
            except (FileNotFoundError, NotADirectoryError):
                self.legend_path = None
                self.vois = np.unique(self.arr).astype('uint16')
                self.vois = self.vois[self.vois != 0]  # exclude the zero VOI (non-segmented area)
                self.legend = pd.Series(self.vois, index=self.vois).to_dict()

        # Generate mask
        self.mask = {}
        for voi in self.vois:
            voi_number = self.legend[voi]
            self.mask[voi] = np.where(self.arr == voi_number)


    def __len__(self):
        return len(self.vois)

    def __str__(self):
            return f'{self.name}, {len(self)} VOIs'

    def find_voi_centers(self):
        voi_centers = []
        for voi in self.vois:
            voi_center = []
            for i in range(self.n_dim):
                voi_center.append(np.mean(self.mask[voi][i]))
            voi_centers.append(voi_center)
        self.voi_centers_df = pd.DataFrame(np.array(voi_centers), index=self.vois, columns=['x', 'y', 'z'])






