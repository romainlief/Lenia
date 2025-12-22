import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, RadioButtons
from matplotlib.figure import Figure
import torch
from ..simulation.const.constantes import *
from const import constantes as CONST
from ..simulation.simulation import Simulation


class SimulationInterface:
    def __init__(self, simulation: Simulation):
        self.simulation = simulation
        self.multi_channel = simulation.multi_channel
        self.x = simulation.x
        self.anim = None
        self.paused = False

    def _enable_resize(self, fig: Figure):
        def _on_resize(event):
            try:
                fig.tight_layout()
            except Exception:
                pass
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("resize_event", _on_resize)

    def zoom_in(self, fig: Figure, factor: float = 1.1):
        ax = fig.axes[0]
        # Sauvegarde des limites initiales au premier appel
        if not hasattr(self, "_initial_xlim"):
            self._initial_xlim = ax.get_xlim()
            self._initial_ylim = ax.get_ylim()

        def _on_zoom(event):
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
            if event.button == "up":
                _on_zoom(event)
            elif event.button == "down":
                _zoom_out(event)

        fig.canvas.mpl_connect("scroll_event", _on_scroll)

    def run(self, num_steps=200):
        if self.multi_channel:
            self._run_multi(num_steps)
        else:
            self._run_single(num_steps)

    def create_species_selector(self, fig: Figure, parent_ax=None):
        """Crée et retourne un sélecteur d'espèce (RadioButtons) et son axe.
        Si parent_ax est fourni, le radio est incrusté dans cet axe (panneau latéral).
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
        radio_species = RadioButtons(ax_species, CONST.AVAILABLE_SPECIES, active=active_index)
        ax_species.set_visible(False)

        def on_species_selected(label):
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

    def _run_single(self, num_steps):
        fig = plt.figure(figsize=(11, 6))
        gs = fig.add_gridspec(
            2, 2,
            height_ratios=[12, 1],
            width_ratios=[8, 3],
            left=0.04, right=0.98, top=0.94, bottom=0.08,
            wspace=0.25, hspace=0.3
        )

        # Simulation
        ax_sim = fig.add_subplot(gs[0, 0])
        # Affichage initial 2D (moyenne si une liste est fournie)
        if isinstance(self.simulation.x, list):
            _disp = torch.mean(torch.stack(self.simulation.x), dim=0)
        else:
            _disp = self.simulation.x
        img = ax_sim.imshow(_disp.cpu().numpy(), cmap="inferno", interpolation="bicubic")
        ax_sim.set_title("Lenia")
        ax_sim.set_xticks([])
        ax_sim.set_yticks([])

        # Panneau latéral: espèces + colormap (colormap plus compacte)
        side_gs = gs[0, 1].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.35)
        ax_species = fig.add_subplot(side_gs[0])
        ax_cmap = fig.add_subplot(side_gs[1])
        for ax in (ax_species, ax_cmap):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(True)
        ax_species.set_title("Species", fontsize=11)
        ax_cmap.set_title("Colormap", fontsize=11)

        # Radio species
        active_species = CONST.AVAILABLE_SPECIES.index(CONST.CURRENT_SPECIES) if CONST.CURRENT_SPECIES in CONST.AVAILABLE_SPECIES else (1 if len(CONST.AVAILABLE_SPECIES) > 1 else 0)
        radio_species = RadioButtons(ax_species, CONST.AVAILABLE_SPECIES, active=active_species)

        # Radio colormap (2e par défaut si dispo)
        default_cmap_index = 1 if len(cmap_list) > 1 else 0
        radio_cmap = RadioButtons(ax_cmap, cmap_list, active=default_cmap_index)

        # Améliore cliquabilité
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

        # Animation
        def update(_):
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
        self.anim = animation.FuncAnimation(fig, update, frames=num_steps, interval=20, blit=False)

        # Callbacks
        def change_species(label):
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
                plt.close(fig)
            except Exception:
                pass
            self.run()

        def change_cmap(label):
            img.set_cmap(label)
            fig.canvas.draw_idle()

        radio_species.on_clicked(change_species)
        radio_cmap.on_clicked(change_cmap)

        # Boutons bas
        bottom_gs = gs[1, :].subgridspec(1, 5)

        def make_button(col, text):
            ax = fig.add_subplot(bottom_gs[0, col])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            return Button(ax, text)

        bpause = make_button(0, STR_BUTTON_PAUSE)
        bresume = make_button(1, STR_BUTTON_RESUME)
        breset = make_button(2, STR_BUTTON_RESET)

        def on_pause(_):
            self.paused = True
            if self.anim is not None:
                self.anim.event_source.stop()

        def on_resume(_):
            self.paused = False
            if self.anim is not None:
                self.anim.event_source.start()

        bpause.on_clicked(on_pause)
        bresume.on_clicked(on_resume)

        def reset(_):
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

    def _run_multi(self, num_steps):
        fig = plt.figure(figsize=(11, 6))
        gs = fig.add_gridspec(
            2, 2,
            height_ratios=[12, 1],
            width_ratios=[8, 3],
            left=0.04, right=0.98, top=0.94, bottom=0.08,
            wspace=0.25, hspace=0.3
        )

        ax_sim = fig.add_subplot(gs[0, 0])
        _x_list = self.simulation.x if isinstance(self.simulation.x, list) else [self.simulation.x]
        rgb = torch.stack(_x_list, dim=2).cpu().numpy()
        im = ax_sim.imshow(rgb, interpolation="bicubic")
        ax_sim.axis("off")
        ax_sim.set_title("Lenia Multi-Channel")

        ax_species = fig.add_subplot(gs[0, 1])
        ax_species.set_title("Species", fontsize=11)
        ax_species.set_xticks([])
        ax_species.set_yticks([])

        radio_species = RadioButtons(ax_species, CONST.AVAILABLE_SPECIES, active=0)
        for c in getattr(radio_species, "circles", []):
            try:
                c.set_radius(0.06)
            except Exception:
                pass

        def change_species(label):
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

        def update(_):
            if self.paused:
                return (im,)
            result = self.simulation.filtre.evolve_lenia(self.simulation.x)
            self.simulation.x = result if isinstance(result, list) else [result]
            rgb = torch.stack(self.simulation.x, dim=2).cpu().numpy()
            im.set_array(rgb)
            return (im,)

        self.anim = animation.FuncAnimation(fig, update, frames=num_steps, interval=40)

        # Bas: boutons (pause/resume/reset)
        bottom_gs = gs[1, :].subgridspec(1, 5)
        def make_button(col, text):
            ax = fig.add_subplot(bottom_gs[0, col])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            return Button(ax, text)

        bpause = make_button(0, STR_BUTTON_PAUSE)
        bresume = make_button(1, STR_BUTTON_RESUME)
        breset = make_button(2, STR_BUTTON_RESET)

        def on_pause_mc(_):
            self.paused = True
            if self.anim is not None:
                self.anim.event_source.stop()

        def on_resume_mc(_):
            self.paused = False
            if self.anim is not None:
                self.anim.event_source.start()

        bpause.on_clicked(on_pause_mc)
        bresume.on_clicked(on_resume_mc)

        def reset(_):
            self.simulation.reset()
            self.paused = False
            _x_list2 = self.simulation.x if isinstance(self.simulation.x, list) else [self.simulation.x]
            rgb = torch.stack(_x_list2, dim=2).cpu().numpy()
            im.set_array(rgb)
            fig.canvas.draw_idle()
        breset.on_clicked(reset)

        self.zoom_in(fig)
        plt.show()

    # create_buttons supprimé: boutons construits inline
