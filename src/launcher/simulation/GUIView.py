import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
from matplotlib.figure import Figure
from dataclasses import dataclass
from typing import Any
import torch
from .const.constantes import *
from const import constantes as CONST


@dataclass
class SingleComponents:
    """
    Components of the single-channel GUI view.
    """
    fig: Figure
    img: Any
    radio_species: RadioButtons
    radio_cmap: RadioButtons
    bpause: Button
    bresume: Button
    breset: Button


@dataclass
class MultiComponents:
    """
    Components of the multi-channel GUI view.
    """
    fig: Figure
    im: Any
    radio_species: RadioButtons
    bpause: Button
    bresume: Button
    breset: Button


def build_single_view(simulation) -> SingleComponents:
    """
    Builds the single-channel GUI view.

    Args:
        simulation (Simulation): The simulation instance.

    Returns:
        SingleComponents: The components of the single-channel GUI view.
    """
    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[12, 1],
        width_ratios=[8, 3],
        left=0.04,
        right=0.98,
        top=0.94,
        bottom=0.08,
        wspace=0.25,
        hspace=0.3,
    )

    # Simulation axes
    ax_sim = fig.add_subplot(gs[0, 0])
    if isinstance(simulation.x, list):
        _disp = torch.mean(torch.stack(simulation.x), dim=0)
    else:
        _disp = simulation.x
    img = ax_sim.imshow(_disp.cpu().numpy(), cmap=BASE_COLORMAP, interpolation="bicubic")
    ax_sim.set_title(SIMULATION_TITLE, fontsize=11)
    ax_sim.set_xticks([])
    ax_sim.set_yticks([])

    # Side panel: species + colormap
    side_gs = gs[0, 1].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.35)
    ax_species = fig.add_subplot(side_gs[0])
    ax_cmap = fig.add_subplot(side_gs[1])
    for ax in (ax_species, ax_cmap):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(True)
    ax_species.set_title(SPECIES_LABEL, fontsize=11)
    ax_cmap.set_title(BUTTON_CMAP, fontsize=11)

    active_species = (
        CONST.AVAILABLE_SPECIES.index(CONST.CURRENT_SPECIES)
        if CONST.CURRENT_SPECIES in CONST.AVAILABLE_SPECIES
        else (1 if len(CONST.AVAILABLE_SPECIES) > 1 else 0)
    )
    radio_species = RadioButtons(
        ax_species, CONST.AVAILABLE_SPECIES, active=active_species
    )

    default_cmap_index = 1 if len(cmap_list) > 1 else 0
    radio_cmap = RadioButtons(ax_cmap, cmap_list, active=default_cmap_index)

    # Bottom buttons
    bottom_gs = gs[1, :].subgridspec(1, 5)

    def make_button(col: int, text: str) -> Button:
        """
        Creates a button in the specified column with the given text.

        Args:
            col (int): The column index where the button will be placed.
            text (str): The text label for the button.

        Returns:
            Button: The created button instance.
        """
        ax = fig.add_subplot(bottom_gs[0, col])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        return Button(ax, text)

    bpause = make_button(0, BUTTON_PAUSE)
    bresume = make_button(1, BUTTON_RESUME)
    breset = make_button(2, BUTTON_RESET)

    # Improve clickability
    for radio in (radio_species, radio_cmap):
        for c in getattr(radio, "circles", []):
            try:
                c.set_radius(0.06)
            except Exception:
                pass
        for label in getattr(radio, "labels", []):
            try:
                label.set_fontsize(10)
            except Exception:
                pass

    return SingleComponents(
        fig, img, radio_species, radio_cmap, bpause, bresume, breset
    )


def build_multi_view(simulation) -> MultiComponents:
    """
    Builds the multi-channel GUI view.

    Args:
        simulation (Simulation): The simulation instance.

    Returns:
        MultiComponents: The components of the multi-channel GUI view.
    """
    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[12, 1],
        width_ratios=[8, 3],
        left=0.04,
        right=0.98,
        top=0.94,
        bottom=0.08,
        wspace=0.25,
        hspace=0.3,
    )

    # Simulation axes
    ax_sim = fig.add_subplot(gs[0, 0])
    _x_list = simulation.x if isinstance(simulation.x, list) else [simulation.x]
    rgb = torch.stack(_x_list, dim=2).cpu().numpy()
    im = ax_sim.imshow(rgb, interpolation="bicubic")
    ax_sim.axis("off")
    ax_sim.set_title(SIMULATION_TITLE + " Multi-Channel", fontsize=11)

    # Side panel: species top, colormap hidden but space preserved
    side_gs = gs[0, 1].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.35)
    ax_species = fig.add_subplot(side_gs[0])
    ax_cmap = fig.add_subplot(side_gs[1])
    for ax in (ax_species, ax_cmap):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(True)
    ax_species.set_title(SPECIES_LABEL, fontsize=11)
    ax_cmap.set_axis_off()

    active_species = (
        CONST.AVAILABLE_SPECIES.index(CONST.CURRENT_SPECIES)
        if CONST.CURRENT_SPECIES in CONST.AVAILABLE_SPECIES
        else (1 if len(CONST.AVAILABLE_SPECIES) > 1 else 0)
    )
    radio_species = RadioButtons(
        ax_species, CONST.AVAILABLE_SPECIES, active=active_species
    )
    for c in getattr(radio_species, "circles", []):
        try:
            c.set_radius(0.06)
        except Exception:
            pass

    # Bottom buttons
    bottom_gs = gs[1, :].subgridspec(1, 5)

    def make_button(col, text):
        ax = fig.add_subplot(bottom_gs[0, col])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        return Button(ax, text)

    bpause = make_button(0, BUTTON_PAUSE)
    bresume = make_button(1, BUTTON_RESUME)
    breset = make_button(2, BUTTON_RESET)

    return MultiComponents(fig, im, radio_species, bpause, bresume, breset)
