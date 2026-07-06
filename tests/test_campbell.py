"""
Campbell-diagram building blocks: the angular-window slicer and the
(RPM-bin × order) accumulator grid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nvh_pipeline import campbell

# viz pulls in altair; skip its spec tests cleanly if altair isn't installed.
pytest.importorskip('altair')
from nvh_pipeline import viz  # noqa: E402


class TestAngularWindows:
    def test_window_count_and_spans(self):
        # 6 revolutions, 2-rev windows, 1-rev hop → starts at 0,1,2,3,4 → 5 windows.
        phi = np.linspace(0, 6, 6000)
        wins = list(campbell.angular_windows(phi, window_rev=2.0, hop_rev=1.0))
        assert len(wins) == 5
        for i0, i1 in wins:
            span = phi[i1 - 1] - phi[i0]
            assert 1.9 <= span <= 2.1

    def test_too_short_stroke_yields_nothing(self):
        phi = np.linspace(0, 1.0, 1000)        # shorter than the window
        assert list(campbell.angular_windows(phi, 2.0, 1.0)) == []

    def test_min_samples_skips_sparse_windows(self):
        phi = np.linspace(0, 4, 40)            # only 10 samples/rev
        wins = list(campbell.angular_windows(phi, 1.0, 1.0, min_samples=100))
        assert wins == []


class TestCampbellGrid:
    def test_mean_per_rpm_bin(self):
        orders = campbell.order_grid(5.0, 1.0)         # 0,1,2,3,4,5
        g = campbell.CampbellGrid(orders, rpm_bin=50.0)
        # Two windows in the same 100–150 RPM bin → mean of their spectra.
        g.add('metal', 120.0, np.full(orders.size, 2.0))
        g.add('metal', 130.0, np.full(orders.size, 4.0))
        # One window in a different bin / material.
        g.add('plastic', 800.0, np.full(orders.size, 1.0))
        df = g.to_long_df()
        assert set(df.columns) == {'material', 'rpm', 'order', 'amplitude'}
        metal_bin = df[(df['material'] == 'metal') & (df['rpm'].between(100, 150))]
        assert np.allclose(metal_bin['amplitude'].unique(), [3.0])   # (2+4)/2
        # Distinct RPM bins exist for each material.
        assert df['rpm'].nunique() >= 2
        assert set(df['material']) == {'metal', 'plastic'}

    def test_rejects_bad_input(self):
        orders = campbell.order_grid(3.0, 1.0)
        g = campbell.CampbellGrid(orders, rpm_bin=50.0)
        g.add('metal', np.nan, np.ones(orders.size))          # bad rpm
        g.add('metal', 100.0, np.ones(orders.size - 1))       # wrong shape
        g.add(None, 100.0, np.ones(orders.size))              # no material
        assert g.to_long_df().empty


def _tooltip_list(enc: dict) -> list:
    tt = enc.get('tooltip', [])
    return tt if isinstance(tt, list) else [tt]


class TestCampbellHeatmapSpec:
    """The Campbell heatmap must tile a real linear RPM axis and its callouts must
    match the hovered cell — the correctness fix for the reported 'whole spectrum
    shown but a single RPM in every callout' defect."""

    def _spec_for(self, tmp_path, cm: pd.DataFrame):
        csv = tmp_path / 'camp.csv'
        cm.to_csv(csv, index=False)
        v = viz.StageView('order', 'Order')
        viz._campbell_heatmap(v, str(tmp_path), str(csv), 'cap', x_title='order')
        assert v.charts, 'expected a Campbell chart'
        return v, v.charts[0][1].to_dict()

    def test_rpm_axis_binned_and_tooltip_matches_cell(self, tmp_path):
        rows = [{'material': 'signal', 'rpm': rpm, 'order': o,
                 'amplitude': rpm * o}
                for rpm in (100, 150, 200, 250, 300) for o in (1.0, 2.0, 3.0)]
        _, spec = self._spec_for(tmp_path, pd.DataFrame(rows))
        enc = spec['encoding']                     # single material -> flat spec
        # RPM (y) is binned so rects tile a linear axis instead of 5px slivers.
        assert 'bin' in enc['y'], enc['y']
        tt = _tooltip_list(enc)
        # Raw 'material' must NOT be a tooltip field (it broke the aggregation).
        assert all(t.get('field') != 'material' for t in tt), tt
        # Amplitude callout is aggregated -> matches the coloured cell.
        amp = [t for t in tt if t.get('field') == 'amplitude']
        assert amp and amp[0].get('aggregate') == 'max', amp
        # rpm / order callouts are binned -> the callout is the hovered cell.
        for f in ('rpm', 'order'):
            ft = [t for t in tt if t.get('field') == f]
            assert ft and 'bin' in ft[0], (f, ft)

    def test_two_materials_are_faceted_never_merged(self, tmp_path):
        rows = [{'material': mat, 'rpm': rpm, 'order': 1.0, 'amplitude': 1.0}
                for mat in ('metal', 'plastic') for rpm in (100, 150, 200)]
        _, spec = self._spec_for(tmp_path, pd.DataFrame(rows))
        assert 'facet' in spec, \
            'metal/plastic must be faceted so their cells can never be merged'


class TestCampbellResolutionNote:
    """The honesty note must be driven by the ACTUAL measured-RPM bin count, so a
    genuine run-up reads as a waterfall while a sparse/degenerate strip is flagged
    — never assuming a fixed 3–5 rev stroke."""

    def _note(self, rpms, rpm_bin=50.0):
        cm = pd.DataFrame({'material': ['signal'] * len(rpms), 'rpm': rpms,
                           'order': [1.0] * len(rpms),
                           'amplitude': [1.0] * len(rpms)})
        v = viz.StageView('order', 'Order')
        viz._campbell_resolution_note(v, cm, rpm_bin)
        assert v.notes
        return v.notes[-1]

    def test_single_speed_flags_snapshot(self):
        note = self._note([125, 125, 125])
        assert note.startswith('⚠') and 'snapshot' in note.lower()

    def test_sparse_clustered_flags_overview(self):
        # Two clusters (~100 and ~2300 RPM) with a large empty gap -> low coverage.
        note = self._note([100, 125, 2300, 2325], rpm_bin=50.0)
        assert note.startswith('⚠') and 'overview' in note.lower()

    def test_dense_sweep_is_factual_not_warned(self):
        note = self._note(list(range(100, 600, 50)), rpm_bin=50.0)
        assert not note.startswith('⚠')
        assert 'measured-RPM bins' in note
