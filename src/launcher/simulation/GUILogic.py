import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, RadioButtons
from matplotlib.figure import Figure
import torch
from .const.constantes import *
from const import constantes as CONST
from .simulation import Simulation
from .GUIView import (
    build_single_view,
    build_multi_view,
    SingleComponents,
    MultiComponents,
)


class GUILogic:
    """
    Class that manages the GUI logic for the Lenia simulation.
    """
    def __init__(self, simulation: Simulation):
        """
        Initialize the GUILogic with a simulation instance.

        Args:
            simulation (Simulation): The simulation instance to manage.
        """
        self.simulation = simulation
        self.multi_channel = simulation.multi_channel
        self.x = simulation.x
        self.anim = None
        self.paused = False

    def _enable_resize(self, fig: Figure):
        """
        Enable dynamic resizing of the figure.

        Args:
            fig (Figure): The figure to enable dynamic resizing for.
        """
        def _on_resize(event):
            """
            Callback for resize event to adjust layout.
            """
            try:
                fig.tight_layout()
            except Exception:
                pass
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("resize_event", func=_on_resize)

    def zoom_in(self, fig: Figure, factor: float = 1.1):
        """
        Enable zooming in and out with the mouse scroll wheel.

        Args:
            fig (Figure): The figure to enable zooming for.
            factor (float, optional): The zoom factor. Defaults to 1.1.
        """
        ax = fig.axes[0]
        # Sauvegarde des limites initiales au premier appel
        if not hasattr(self, "_initial_xlim"):
            self._initial_xlim = ax.get_xlim()
            self._initial_ylim = ax.get_ylim()

        def _on_zoom(event):
            """
            Callback for zooming in.
            """
            ax = fig.axes[0]
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            x_center = (xlim[0] + xlim[1]) / 2
            y_center = (ylim[0] + ylim[1]) / 2
            x_range = (xlim[1] - xlim[0]) / factor
            y_range = (ylim[1] - ylim[0]) / factor
            ax.set_xlim(x_center - x_range / 2, x_center + x_range / 2)
            ax.set_ylim(y_center - y_range / 2, y_center + y_range / 2)
            fig.canvas.draw_idle()

        def _zoom_out(event):
            """
            Callback for zooming out.
            """
            ax = fig.axes[0]
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            # Si on est déjà à l'état initial, ne rien faire
            if (
                abs(xlim[0] - self._initial_xlim[0]) < 1e-6
                and abs(xlim[1] - self._initial_xlim[1]) < 1e-6
                and abs(ylim[0] - self._initial_ylim[0]) < 1e-6
                and abs(ylim[1] - self._initial_ylim[1]) < 1e-6
            ):
                return
            x_center = (xlim[0] + xlim[1]) / 2
            y_center = (ylim[0] + ylim[1]) / 2
            x_range = (xlim[1] - xlim[0]) * factor
            y_range = (ylim[1] - ylim[0]) * factor
            new_xlim = (x_center - x_range / 2, x_center + x_range / 2)
            new_ylim = (y_center - y_range / 2, y_center + y_range / 2)
            # Si le nouveau zoom dépasse l'initial, on remet l'initial
            if (
                new_xlim[0] < self._initial_xlim[0]
                or new_xlim[1] > self._initial_xlim[1]
                or new_ylim[0] < self._initial_ylim[0]
                or new_ylim[1] > self._initial_ylim[1]
            ):
                ax.set_xlim(self._initial_xlim)
                ax.set_ylim(self._initial_ylim)
            else:
                ax.set_xlim(new_xlim)
                ax.set_ylim(new_ylim)
            fig.canvas.draw_idle()

        def _on_scroll(event):
            """
            Callback for scroll event to zoom in/out.
            """
            if event.button == "up":
                _on_zoom(event)
            elif event.button == "down":
                _zoom_out(event)

        fig.canvas.mpl_connect("scroll_event", _on_scroll)

    def run(self, num_steps=200):
        """
        Run the GUI simulation.

        Args:
            num_steps (int, optional): The number of steps to run the simulation. Defaults to 200.
        """
        if self.multi_channel:
            self._run_multi(num_steps)
        else:
            self._run_single(num_steps)

    def create_species_selector(self, fig: Figure, parent_ax=None):
        """
        Create a species selector radio button.
        """
        if parent_ax is None:
            # Petit panneau flottant avec marges fixes
            ax_species = plt.axes((0.02, 0.30, 0.20, 0.50))
        else:
            # Inset centré avec marges fixes pour stabilité
            ax_species = parent_ax.inset_axes((0.05, 0.10, 0.90, 0.80))
        try:
            active_index = CONST.AVAILABLE_SPECIES.index(CONST.CURRENT_SPECIES)
        except Exception:
            # Par défaut, 1ère option si indisponible
            active_index = 0
        radio_species = RadioButtons(
            ax_species, CONST.AVAILABLE_SPECIES, active=active_index
        )
        ax_species.set_visible(False)

        def on_species_selected(label):
            """
            Callback when a species is selected.

            Args:
                label (str): The selected species label.
            """
            new_kernel_type = CONST.set_species_parameters(label)
            self.simulation.reinitialize_species(new_kernel_type)
            ax_species.set_visible(False)
            try:
                plt.close(fig)
            except Exception:
                pass
            self.multi_channel = self.simulation.multi_channel
            self.run()

        radio_species.on_clicked(on_species_selected)
        return radio_species, ax_species

    def _run_single(self, num_steps) -> None:
        """
        Run the single-channel GUI simulation.

        Args:
            num_steps (int): The number of steps to run the simulation.
        """
        comps: SingleComponents = build_single_view(self.simulation)
        fig = comps.fig
        img = comps.img
        radio_species = comps.radio_species
        radio_cmap = comps.radio_cmap
        bpause, bresume, breset = comps.bpause, comps.bresume, comps.breset

        # Animation
        def update(_) -> tuple:
            """
            Callback for updating the simulation frame.
            
            Returns:
                tuple: Updated image data.
            """
            if self.paused:
                return (img,)
            result = self.simulation.filtre.evolve_lenia(self.simulation.x)
            if isinstance(result, list):
                self.simulation.x = [r.clone() for r in result]
                display = torch.mean(torch.stack(self.simulation.x), dim=0)
            else:
                self.simulation.x = result
                display = result
            img.set_data(display.cpu().numpy())
            return (img,)

        # blit=False pour stabilité sur macOS backends
        self.anim = animation.FuncAnimation(
            fig, update, frames=num_steps, interval=20, blit=False
        )

        # Callbacks
        def change_species(label) -> None:
            """
            Callback for changing species.

            Args:
                label (str): The selected species label.
            """
            # Arrête et libère proprement l'animation avant fermeture
            if self.anim is not None:
                try:
                    self.anim.event_source.stop()
                except Exception:
                    pass
                self.anim = None
            kernel = CONST.set_species_parameters(label)
            self.simulation.reinitialize_species(kernel)
            # Met à jour le mode (mono/multi) avant de relancer
            self.multi_channel = self.simulation.multi_channel
            try:
                print("ici")
                plt.close(fig)
            except Exception:
                pass
            self.run()

        def change_cmap(label) -> None:
            """
            Callback for changing the colormap.
            Args:
                label (str): The selected colormap label.
            """
            img.set_cmap(label)
            fig.canvas.draw_idle()

        radio_species.on_clicked(change_species)
        radio_cmap.on_clicked(change_cmap)

        # Boutons bas
        # Boutons déjà construits par le builder

        def on_pause(_) -> None:
            """
            Callback for pausing the simulation.
            """
            self.paused = True
            if self.anim is not None:
                self.anim.event_source.stop()

        def on_resume(_) -> None:
            """
            Callback for resuming the simulation.
            """
            self.paused = False
            if self.anim is not None:
                self.anim.event_source.start()

        bpause.on_clicked(on_pause)
        bresume.on_clicked(on_resume)

        def reset(_) -> None:
            """
            Callback for resetting the simulation.
            """
            self.simulation.reset()
            self.paused = False
            if isinstance(self.simulation.x, list):
                disp = torch.mean(torch.stack(self.simulation.x), dim=0)
            else:
                disp = self.simulation.x
            img.set_data(disp.cpu().numpy())
            fig.canvas.draw_idle()

        breset.on_clicked(reset)

        self.zoom_in(fig)
        plt.show()

    def _run_multi(self, num_steps) -> None:
        """
        Run the multi-channel GUI simulation.

        Args:
            num_steps (int): number of steps to run the simulation.
        """
        comps: MultiComponents = build_multi_view(self.simulation)
        fig = comps.fig
        im = comps.im
        radio_species = comps.radio_species
        bpause, bresume, breset = comps.bpause, comps.bresume, comps.breset

        def change_species(label) -> None:
            """
            Callback for changing species.

            Args:
                label (str): The selected species label.
            """
            if self.anim is not None:
                try:
                    self.anim.event_source.stop()
                except Exception:
                    pass
                self.anim = None
            kernel = CONST.set_species_parameters(label)
            self.simulation.reinitialize_species(kernel)
            # Met à jour le mode (mono/multi) avant de relancer
            self.multi_channel = self.simulation.multi_channel
            try:
                plt.close(fig)
            except Exception:
                pass
            self.paused = False
            self.run()

        radio_species.on_clicked(change_species)

        def update(_) -> tuple:
            """
            Callback for updating the simulation frame.

            Returns:
                tuple: Updated image data.
            """
            if self.paused:
                return (im,)
            result = self.simulation.filtre.evolve_lenia(self.simulation.x)
            self.simulation.x = result if isinstance(result, list) else [result]
            rgb = torch.stack(self.simulation.x, dim=2).cpu().numpy()
            im.set_array(rgb)
            return (im,)

        # blit=False pour stabilité sur macOS
        self.anim = animation.FuncAnimation(
            fig, update, frames=num_steps, interval=40, blit=False
        )

        # Bas: boutons (pause/resume/reset)
        # Boutons déjà construits par le builder

        def on_pause_mc(_) -> None:
            """
            Callback for pausing the simulation.
            """
            self.paused = True
            if self.anim is not None:
                self.anim.event_source.stop()

        def on_resume_mc(_) -> None:
            """
            Callback for resuming the simulation.
            """
            self.paused = False
            if self.anim is not None:
                self.anim.event_source.start()

        bpause.on_clicked(on_pause_mc)
        bresume.on_clicked(on_resume_mc)

        def reset(_) -> None:
            """
            Callback for resetting the simulation.
            """
            self.simulation.reset()
            self.paused = False
            _x_list2 = (
                self.simulation.x
                if isinstance(self.simulation.x, list)
                else [self.simulation.x]
            )
            rgb = torch.stack(_x_list2, dim=2).cpu().numpy()
            im.set_array(rgb)
            fig.canvas.draw_idle()

        breset.on_clicked(reset)

        self.zoom_in(fig)
        plt.show()
