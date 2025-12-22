import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from matplotlib.image import AxesImage
from matplotlib.figure import Figure
import torch

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
            x_center = (xlim[0] + xlim[1]) / 2
            y_center = (ylim[0] + ylim[1]) / 2
            x_range = (xlim[1] - xlim[0]) * factor
            y_range = (ylim[1] - ylim[0]) * factor
            ax.set_xlim(x_center - x_range / 2, x_center + x_range / 2)
            ax.set_ylim(y_center - y_range / 2, y_center + y_range / 2)
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

    def _run(self):
        fig, ax = plt.subplots()
        self.img = ax.imshow(self.simulation.x, cmap="inferno", interpolation="none")
        ax.set_title("Lenia")
        ax.set_xticks([])
        ax.set_yticks([])

        axpause = plt.axes((0.7, 0.01, 0.1, 0.05))
        axresume = plt.axes((0.81, 0.01, 0.1, 0.05))
        axreset = plt.axes((0.59, 0.01, 0.1, 0.05))
        bpause = Button(axpause, "Pause")
        bresume = Button(axresume, "Reprendre")
        breset = Button(axreset, "Reset")

        def update(frame):
            result = self.simulation.filtre.evolve_lenia(self.simulation.x)
            if isinstance(result, list):
                self.simulation.x = [r.clone() for r in result]
                display = torch.mean(torch.stack(self.simulation.x, dim=0), dim=0).cpu().numpy()
            else:
                self.simulation.x = result
                display = self.simulation.x.cpu().numpy()
            self.img.set_data(display)
            return [self.img]

        anim = animation.FuncAnimation(
            fig, update, frames=200, interval=20, blit=True
        )

        def pause(event):
            anim.event_source.stop()
            
        def resume(event):
            anim.event_source.start()
            
        def reset(event):
            if self.simulation.X_raw is not None:
                self.simulation.reset()
            self.img.set_data(self.simulation.x.cpu().numpy()) # type: ignore
            fig.canvas.draw_idle()

        bpause.on_clicked(pause)
        bresume.on_clicked(resume)
        breset.on_clicked(reset)

        self._enable_resize(fig)
        self.zoom_in(fig=fig)
        plt.show()

    def _run_multi(self, num_steps=100, interpolation="bicubic"):
        fig, ax = plt.subplots()
        self.x = [
            torch.as_tensor(x) if not isinstance(x, torch.Tensor) else x for x in self.simulation.x
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
            rgb = torch.stack([self.simulation.x[1], self.simulation.x[2], self.simulation.x[0]], dim=2).cpu().numpy()
            im.set_array(rgb)
            return (im,)

        ani = animation.FuncAnimation(
            fig, update_multi, frames=num_steps, interval=50, blit=False
        )

        axpause = plt.axes((0.7, 0.01, 0.1, 0.05))
        axresume = plt.axes((0.81, 0.01, 0.1, 0.05))
        axreset = plt.axes((0.59, 0.01, 0.1, 0.05))
        bpause = Button(axpause, "Pause")
        bresume = Button(axresume, "Reprendre")
        breset = Button(axreset, "Reset")

        def pause(event):
            ani.event_source.stop()
            
        def resume(event):
            ani.event_source.start()
            
        def reset(event):
            if self.simulation.X_raw is not None:
                self.simulation.reset()
            rgb = torch.stack([self.simulation.x[1], self.simulation.x[2], self.simulation.x[0]], dim=2).cpu().numpy()
            im.set_array(rgb)
            fig.canvas.draw_idle()

        bpause.on_clicked(pause)
        bresume.on_clicked(resume)
        breset.on_clicked(reset)

        self._enable_resize(fig)
        self.zoom_in(fig=fig)
        plt.show()
        