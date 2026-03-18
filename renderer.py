import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from stl import mesh

# =========================================================
# STARFIELD (mplstl6-style, spherical + depth)
# =========================================================

class Starfield:
    def __init__(self, n=4000):
        phi = np.random.uniform(0, 2*np.pi, n)
        costheta = np.random.uniform(-1, 1, n)
        u = np.random.rand(n)

        theta = np.arccos(costheta)
        r = 1.5 + u * 3.5

        self.x = r * np.sin(theta) * np.cos(phi)
        self.y = r * np.sin(theta) * np.sin(phi)
        self.z = r * np.cos(theta)

        self.mag = np.random.rand(n) ** 2
        self.color = np.random.choice([
            [1.0, 1.0, 1.0],
            [1.0, 0.9, 0.8],
            [0.8, 0.9, 1.0]
        ], size=n)

    def render(self, W, H, t=0, animate=True):

        ang = t * 0.2 if animate else 0
        ca, sa = np.cos(ang), np.sin(ang)

        x = self.x * ca - self.z * sa
        z = self.x * sa + self.z * ca
        y = self.y

        fb = np.zeros((H, W, 3), dtype=float)

        proj = 1 / (z + 5)

        px = (x * proj * W/2 + W/2).astype(int)
        py = (y * proj * H/2 + H/2).astype(int)

        mask = (px>=0)&(px<W)&(py>=0)&(py<H)

        px, py = px[mask], py[mask]
        mag = self.mag[mask]
        col = self.color[mask]

        fb[py, px] = col * mag[:,None]

        return fb

# =========================================================
# BASIC SOFTWARE RASTERIZER (mplstl7 core idea)
# =========================================================

class Rasterizer:
    def render(self, tris, norms, color, light_dir, W, H, bg=None, show_edges=False):

        fb = np.zeros((H, W, 3), float) if bg is None else bg.copy()
        zbuf = np.full((H, W), np.inf)

        light_dir = light_dir / np.linalg.norm(light_dir)

        for tri, n in zip(tris, norms):

            pts = tri.copy()

            z = pts[:,2] + 5
            proj = 1 / z

            sx = (pts[:,0] * proj * W/2 + W/2)
            sy = (pts[:,1] * proj * H/2 + H/2)

            pts2d = np.stack([sx, sy], axis=1)

            minx = int(max(0, np.floor(sx.min())))
            maxx = int(min(W-1, np.ceil(sx.max())))
            miny = int(max(0, np.floor(sy.min())))
            maxy = int(min(H-1, np.ceil(sy.max())))

            intensity = max(0, np.dot(n, light_dir))
            col = color * intensity

            # depth fade (new)
            avg_z = np.mean(z)
            fade = np.clip(1.2 - avg_z*0.15, 0.6, 1.0)
            col *= fade

            for i in range(minx, maxx):
                for j in range(miny, maxy):

                    if self.point_in_tri(i, j, pts2d):

                        if avg_z < zbuf[j,i]:
                            zbuf[j,i] = avg_z
                            fb[j,i] = col

            if show_edges:
                self.draw_edges(fb, pts2d)

        return fb

    def point_in_tri(self, x, y, tri):
        def sign(p1, p2, p3):
            return (p1[0]-p3[0])*(p2[1]-p3[1]) - (p2[0]-p3[0])*(p1[1]-p3[1])

        b1 = sign([x,y], tri[0], tri[1]) < 0
        b2 = sign([x,y], tri[1], tri[2]) < 0
        b3 = sign([x,y], tri[2], tri[0]) < 0

        return (b1 == b2) and (b2 == b3)

    def draw_edges(self, fb, pts):
        pts = pts.astype(int)
        for i in range(3):
            p1 = pts[i]
            p2 = pts[(i+1)%3]
            self.line(fb, p1, p2)

    def line(self, fb, p1, p2):
        x1,y1 = p1
        x2,y2 = p2

        dx = abs(x2-x1)
        dy = abs(y2-y1)

        x,y = x1,y1
        sx = 1 if x2>x1 else -1
        sy = 1 if y2>y1 else -1

        if dx > dy:
            err = dx/2
            while x != x2:
                if 0<=x<fb.shape[1] and 0<=y<fb.shape[0]:
                    fb[y,x] = [1,1,1]
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy/2
            while y != y2:
                if 0<=x<fb.shape[1] and 0<=y<fb.shape[0]:
                    fb[y,x] = [1,1,1]
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy

# =========================================================
# MAIN VIEWER (mplstl7 structure simplified)
# =========================================================

class Viewer:
    def __init__(self, stl_file):

        m = mesh.Mesh.from_file(stl_file)
        self.tris = m.vectors.copy()

        center = self.tris.reshape(-1,3).mean(axis=0)
        self.tris -= center

        self.norms = self.compute_normals(self.tris)

        self.starfield = Starfield()
        self.rasterizer = Rasterizer()

        self.color = np.array([0.8,0.8,0.9])
        self.light_dir = np.array([1,1,1])

        self.rot = 0
        self.animate = True
        self.show_edges = False
        self.show_stars = True

        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(np.zeros((600,800,3)))
        self.ax.axis("off")

        self.anim = FuncAnimation(self.fig, self.update, interval=30)

    def compute_normals(self, tri):
        v1 = tri[:,1]-tri[:,0]
        v2 = tri[:,2]-tri[:,0]
        n = np.cross(v1,v2)
        n /= np.linalg.norm(n,axis=1)[:,None]
        return n

    def rotate(self):
        a = self.rot
        R = np.array([
            [np.cos(a),0,np.sin(a)],
            [0,1,0],
            [-np.sin(a),0,np.cos(a)]
        ])
        return self.tris @ R.T

    def update(self, frame):

        if self.animate:
            self.rot += 0.02

        tris = self.rotate()

        W, H = 800, 600

        bg = self.starfield.render(W, H, self.rot) if self.show_stars else None

        fb = self.rasterizer.render(
            tris,
            self.norms,
            self.color,
            self.light_dir,
            W,
            H,
            bg=bg,
            show_edges=self.show_edges
        )

        self.im.set_data(fb)
        return self.im

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--edges", action="store_true")
    parser.add_argument("--no-stars", action="store_true")

    args = parser.parse_args()

    v = Viewer(args.file)

    v.show_edges = args.edges
    v.show_stars = not args.no_stars

    plt.show()