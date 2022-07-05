from apps.shared.generic_3d_viewer_app import Generic3DViewerApp


class BlockVisualizerApp(Generic3DViewerApp):
    def __init__(self):
        super.__init__()
        # set up camera

        self.offset_cam = (0.2562770766576, 0.13962609403401335, -0.2113334598208764)
        self.camera.SetPosition(self.offset_cam[0], self.offset_cam[1], self.offset_cam[2])
        self.camera.SetPosition(0, 0, -1)
        self.camera.SetViewUp(0, 1.0, 0)
        self.camera.SetFocalPoint(0, 0, 1.5)
        self.camera.SetClippingRange(0.01, 10.0)
        self.render_window.Render()


def visualize_blocks():
    app = Generic3DViewerApp()
    app.launch()


if __name__ == "__main__":
    visualize_blocks()
