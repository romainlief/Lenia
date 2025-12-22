import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider, RadioButtons
from matplotlib.figure import Figure
import torch
from ..simulation.const.constantes import *
from const.constantes import channel_count
from ..simulation.simulation import Simulation


class SimulationInterface:
    def __init__(self, simulation: Simulation):
        self.simulation = simulation
        self.multi_channel = simulation.multi_channel
        self.x = simulation.x

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

    def run(self, num_steps=100, interpolation="bicubic"):
        if self.multi_channel:
            self._run_multi(num_steps=num_steps, interpolation=interpolation)
        else:
            self._run()

    def _run(self, cmap_list=cmap_list):
        fig, ax = plt.subplots()
        self.img = ax.imshow(self.simulation.x, cmap="inferno", interpolation="none")
        ax.set_title("Lenia")
        ax.set_xticks([])
        ax.set_yticks([])

        bpause, bresume, breset, bcolormap = self.create_buttons(fig)

        axcmaps = plt.axes((0.01, 0.01, 0.15, 0.18))
        radio = RadioButtons(axcmaps, cmap_list, active=0)
        axcmaps.set_visible(False)  # Toujours invisible au départ

        def update(frame):
            result = self.simulation.filtre.evolve_lenia(self.simulation.x)
            if isinstance(result, list):
                self.simulation.x = [r.clone() for r in result]
                display = (
                    torch.mean(torch.stack(self.simulation.x, dim=0), dim=0)
                    .cpu()
                    .numpy()
                )
            else:
                self.simulation.x = result
                display = self.simulation.x.cpu().numpy()
            self.img.set_data(display)
            return [self.img]

        anim = animation.FuncAnimation(fig, update, frames=200, interval=20, blit=True)

        def pause(event):
            anim.event_source.stop()

        def resume(event):
            anim.event_source.start()

        def reset(event):
            if self.simulation.X_raw is not None:
                self.simulation.reset()
            self.img.set_data(self.simulation.x.cpu().numpy())  # type: ignore
            fig.canvas.draw_idle()

        def show_cmap(event):
            axcmaps.set_visible(True)
            fig.canvas.draw_idle()

        def change_cmap(label):
            axcmaps.set_visible(False)  # Masque après sélection
            self.img.set_cmap(label)
            fig.canvas.draw_idle()

        bpause.on_clicked(pause)
        bresume.on_clicked(resume)
        breset.on_clicked(reset)
        if channel_count == 1 and bcolormap is not None:
            bcolormap.on_clicked(show_cmap)
            radio.on_clicked(change_cmap)

        self._enable_resize(fig)
        self.zoom_in(fig=fig)
        plt.show()

    def _run_multi(self, num_steps=100, interpolation="bicubic"):
        fig, ax = plt.subplots()
        self.x = [
            torch.as_tensor(x) if not isinstance(x, torch.Tensor) else x
            for x in self.simulation.x
        ]
        im = ax.imshow(
            torch.stack(self.x, dim=2).cpu().numpy(), interpolation=interpolation
        )
        ax.axis("off")
        ax.set_title("Lenia Multi-Channel")

        def update_multi(i):
            result = self.simulation.filtre.evolve_lenia(self.simulation.x)
            if not isinstance(result, list):
                if isinstance(result, torch.Tensor):
                    self.simulation.x = [result] if result.ndim == 2 else list(result)
                else:
                    self.simulation.x = [torch.as_tensor(result)]
            else:
                self.simulation.x = [
                    torch.as_tensor(r) if not isinstance(r, torch.Tensor) else r
                    for r in result
                ]
            rgb = (
                torch.stack(
                    [self.simulation.x[1], self.simulation.x[2], self.simulation.x[0]],
                    dim=2,
                )
                .cpu()
                .numpy()
            )
            im.set_array(rgb)
            return (im,)

        ani = animation.FuncAnimation(
            fig, update_multi, frames=num_steps, interval=50, blit=False
        )

        bpause, bresume, breset, bcolormap = self.create_buttons(fig)

        def pause(event):
            ani.event_source.stop()

        def resume(event):
            ani.event_source.start()

        def reset(event):
            if self.simulation.X_raw is not None:
                self.simulation.reset()
            rgb = (
                torch.stack(
                    [self.simulation.x[1], self.simulation.x[2], self.simulation.x[0]],
                    dim=2,
                )
                .cpu()
                .numpy()
            )
            im.set_array(rgb)
            fig.canvas.draw_idle()

        def change_speed(val):
            try:
                new_interval = max(1, int(50 / val))
                ani.event_source.interval = new_interval
            except Exception:
                pass

        bpause.on_clicked(pause)
        bresume.on_clicked(resume)
        breset.on_clicked(reset)

        self._enable_resize(fig)
        self.zoom_in(fig=fig)
        plt.show()

    def create_buttons(self, fig: Figure, channel_count: int = channel_count):
        axpause = plt.axes((0.7, 0.01, len(STR_BUTTON_PAUSE) * 0.02, 0.05))
        axresume = plt.axes((0.81, 0.01, len(STR_BUTTON_RESUME) * 0.02, 0.05))
        axreset = plt.axes((0.59, 0.01, len(STR_BUTTON_RESET) * 0.02, 0.05))
        bpause = Button(axpause, STR_BUTTON_PAUSE)
        bresume = Button(axresume, STR_BUTTON_RESUME)
        breset = Button(axreset, STR_BUTTON_RESET)

        if channel_count == 1:
            axcolormap = plt.axes((0.38, 0.01, len(STR_BUTTON_CMAP) * 0.02, 0.05))
            bcolormap = Button(axcolormap, STR_BUTTON_CMAP)
            return bpause, bresume, breset, bcolormap
        else:
            return bpause, bresume, breset, None
