"""
Material graphs for the eyes, skin, lashes and overlays (Blender only).

Everything is procedural so the scene has no external texture files: the iris,
sclera and skin maps come from lab/textures.py and are packed into scene.blend.

Per-object custom properties read by the shaders (Attribute node, Object type):
    pupil_ratio        pupil radius / globe radius
    iris_ratio         iris radius / globe radius
    cornea_refraction  0 = thin specular shell, 1 = refracting cornea (IOR 1.376)
"""
import math

import bpy
import numpy as np

from lab import textures


# -- small node-graph helper ----------------------------------------------------
class Graph:
    def __init__(self, material):
        try:
            material.use_nodes = True
        except (AttributeError, TypeError):
            pass
        self.tree = material.node_tree
        self.tree.nodes.clear()
        self.x = 0

    def node(self, kind, **props):
        n = self.tree.nodes.new(kind)
        for key, val in props.items():
            setattr(n, key, val)
        n.location = (self.x, 0)
        self.x += 40
        return n

    def link(self, out, inp):
        self.tree.links.new(out, inp)

    def feed(self, socket, value):
        if value is None:
            return
        if isinstance(value, bpy.types.NodeSocket):
            self.link(value, socket)
        else:
            socket.default_value = value

    @staticmethod
    def pick(sockets, identifier):
        for s in sockets:
            if s.identifier == identifier:
                return s
        return sockets[identifier]

    # scalar maths
    def math(self, op, a, b=None, c=None, clamp=False):
        n = self.node("ShaderNodeMath", operation=op, use_clamp=clamp)
        for i, val in enumerate((a, b, c)):
            self.feed(n.inputs[i], val)
        return n.outputs[0]

    def add(self, a, b): return self.math("ADD", a, b)
    def sub(self, a, b): return self.math("SUBTRACT", a, b)
    def mul(self, a, b): return self.math("MULTIPLY", a, b)
    def div(self, a, b): return self.math("DIVIDE", a, b)

    def vmath(self, op, a, b=None, scale=None):
        n = self.node("ShaderNodeVectorMath", operation=op)
        self.feed(n.inputs[0], a)
        if b is not None:
            self.feed(n.inputs[1], b)
        if scale is not None:
            self.feed(n.inputs["Scale"], scale)
        if op in ("DOT_PRODUCT", "LENGTH", "DISTANCE"):
            return n.outputs["Value"]
        return n.outputs["Vector"]

    def xyz(self, vec):
        n = self.node("ShaderNodeSeparateXYZ")
        self.link(vec, n.inputs[0])
        return n.outputs[0], n.outputs[1], n.outputs[2]

    def vec(self, x, y, z):
        n = self.node("ShaderNodeCombineXYZ")
        for i, val in enumerate((x, y, z)):
            self.feed(n.inputs[i], val)
        return n.outputs[0]

    def remap(self, value, lo, hi, smooth=False, out=(0.0, 1.0)):
        """Map value from [lo, hi] to out, clamped. lo > hi flips the ramp."""
        n = self.node("ShaderNodeMapRange", data_type="FLOAT",
                      interpolation_type="SMOOTHSTEP" if smooth else "LINEAR", clamp=True)
        self.feed(n.inputs["Value"], value)
        self.feed(n.inputs["From Min"], lo)
        self.feed(n.inputs["From Max"], hi)
        self.feed(n.inputs["To Min"], out[0])
        self.feed(n.inputs["To Max"], out[1])
        return n.outputs[0]

    def mix(self, fac, a, b, kind="RGBA"):
        n = self.node("ShaderNodeMix", data_type=kind)
        suffix = {"RGBA": "Color", "FLOAT": "Float", "VECTOR": "Vector"}[kind]
        self.feed(self.pick(n.inputs, "Factor_Float"), fac)
        self.feed(self.pick(n.inputs, f"A_{suffix}"), a)
        self.feed(self.pick(n.inputs, f"B_{suffix}"), b)
        return self.pick(n.outputs, f"Result_{suffix}")

    def attr(self, name):
        n = self.node("ShaderNodeAttribute", attribute_type="OBJECT", attribute_name=name)
        return n.outputs["Fac"]

    def rgb(self, colour, label=None):
        n = self.node("ShaderNodeRGB")
        n.outputs[0].default_value = (*colour, 1.0)
        if label:
            n.label = n.name = label
        return n.outputs[0]

    def image(self, img, vector, extension="REPEAT", projection="FLAT", blend=0.0):
        n = self.node("ShaderNodeTexImage", image=img, extension=extension,
                      interpolation="Cubic", projection=projection)
        n.projection_blend = blend
        self.link(vector, n.inputs["Vector"])
        return n.outputs["Color"]

    def luminance(self, colour):
        n = self.node("ShaderNodeRGBToBW")
        self.link(colour, n.inputs[0])
        return n.outputs[0]

    def output(self, shader):
        n = self.node("ShaderNodeOutputMaterial")
        self.link(shader, n.inputs["Surface"])
        return n


def set_input(node, names, value):
    """Principled sockets were renamed across Blender versions; try each name."""
    for name in names:
        if name in node.inputs:
            node.inputs[name].default_value = value
            return node.inputs[name]
    return None


def socket(node, *names):
    for name in names:
        if name in node.inputs:
            return node.inputs[name]
    raise KeyError(names)


# -- images ------------------------------------------------------------------
def make_image(name, pixels, folder, colorspace="sRGB"):
    """Write an HxWx3 float array to folder/<name>.png, load it and pack it into the .blend."""
    h, w = pixels.shape[:2]
    old = bpy.data.images.get(name)
    if old is not None:
        bpy.data.images.remove(old)
    rgba = textures.with_alpha(pixels) if pixels.shape[2] == 3 else pixels
    if colorspace == "sRGB":
        # arrays are authored in display space; sRGB images store encoded values
        rgba = rgba.copy()
        rgba[..., :3] = np.clip(rgba[..., :3], 0.0, 1.0) ** (1.0 / 2.2)
    tmp = bpy.data.images.new(name + "_tmp", width=w, height=h, alpha=True)
    tmp.pixels.foreach_set(np.ascontiguousarray(rgba, dtype=np.float32).ravel())
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name.lower()}.png"
    tmp.filepath_raw = str(path)
    tmp.file_format = "PNG"
    tmp.save()
    bpy.data.images.remove(tmp)
    img = bpy.data.images.load(str(path))
    img.name = name
    img.colorspace_settings.name = colorspace
    img.pack()
    return img


def replace_pixels(img, pixels):
    """Swap the contents of a packed sRGB image (used for identity changes)."""
    rgba = textures.with_alpha(pixels).copy()
    rgba[..., :3] = np.clip(rgba[..., :3], 0.0, 1.0) ** (1.0 / 2.2)
    img.pixels.foreach_set(np.ascontiguousarray(rgba, dtype=np.float32).ravel())
    img.update()


# -- eye ----------------------------------------------------------------------
def eye_material(name, iris_img, sclera_img):
    """Globe shader driven by object-space angle from the optical axis (local -Y)."""
    mat = bpy.data.materials.new(name)
    g = Graph(mat)
    coords = g.node("ShaderNodeTexCoord").outputs["Object"]
    n = g.vmath("NORMALIZE", coords)
    nx, ny, nz = g.xyz(n)
    theta = g.math("ARCCOSINE", g.mul(ny, -1.0))
    azimuth = g.math("ARCTAN2", nx, nz)
    u = g.add(g.mul(azimuth, 1.0 / (2.0 * math.pi)), 0.5)

    theta_pupil = g.math("ARCSINE", g.attr("pupil_ratio"))
    theta_iris = g.math("ARCSINE", g.attr("iris_ratio"))

    iris_v = g.remap(theta, theta_pupil, theta_iris)
    iris_col = g.image(iris_img, g.vec(u, g.math("MINIMUM", iris_v, 0.998), 0.0), extension="EXTEND")
    sclera_v = g.remap(theta, theta_iris, math.pi / 2 + 0.4)
    sclera_col = g.image(sclera_img, g.vec(u, sclera_v, 0.0), extension="EXTEND")

    limbus = g.remap(theta, g.sub(theta_iris, 0.035), g.add(theta_iris, 0.02), smooth=True)
    pupil_edge = g.remap(theta, g.sub(theta_pupil, 0.012), g.add(theta_pupil, 0.010), smooth=True)

    colour = g.mix(limbus, iris_col, sclera_col)
    colour = g.mix(pupil_edge, (0.003, 0.003, 0.003, 1.0), colour)
    rough = g.mix(limbus, 0.38, 0.16, kind="FLOAT")
    rough = g.mix(pupil_edge, 0.05, rough, kind="FLOAT")

    # stroma relief: iris brightness as a height field, iris only
    relief = g.mul(g.luminance(iris_col), g.sub(1.0, limbus))
    bump = g.node("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35
    bump.inputs["Distance"].default_value = 0.0002
    g.link(relief, bump.inputs["Height"])

    bsdf = g.node("ShaderNodeBsdfPrincipled")
    g.link(colour, socket(bsdf, "Base Color"))
    g.link(rough, socket(bsdf, "Roughness"))
    g.link(bump.outputs[0], socket(bsdf, "Normal"))
    set_input(bsdf, ("Subsurface Weight", "Subsurface"), 0.12)
    set_input(bsdf, ("Subsurface Radius",), (0.8, 0.3, 0.2))
    set_input(bsdf, ("Subsurface Scale",), 0.0015)
    set_input(bsdf, ("Specular IOR Level", "Specular"), 0.5)
    g.output(bsdf.outputs[0])
    return mat


def cornea_material(name, ior):
    """Clear cornea. Default is a thin specular shell (no ray bending, exact ground truth);
    cornea_refraction = 1 switches to a refracting glass body."""
    mat = bpy.data.materials.new(name)
    g = Graph(mat)
    fresnel = g.node("ShaderNodeFresnel")
    fresnel.inputs["IOR"].default_value = ior
    clear = g.node("ShaderNodeBsdfTransparent").outputs[0]
    gloss = g.node("ShaderNodeBsdfGlossy")
    gloss.inputs["Roughness"].default_value = 0.015
    shell = g.node("ShaderNodeMixShader")
    g.link(fresnel.outputs[0], shell.inputs[0])
    g.link(clear, shell.inputs[1])
    g.link(gloss.outputs[0], shell.inputs[2])

    glass = g.node("ShaderNodeBsdfGlass")
    glass.inputs["IOR"].default_value = ior
    glass.inputs["Roughness"].default_value = 0.0

    body = g.node("ShaderNodeMixShader")
    g.link(g.attr("cornea_refraction"), body.inputs[0])
    g.link(shell.outputs[0], body.inputs[1])
    g.link(glass.outputs[0], body.inputs[2])

    # never let the cornea shadow the iris
    path = g.node("ShaderNodeLightPath")
    final = g.node("ShaderNodeMixShader")
    g.link(path.outputs["Is Shadow Ray"], final.inputs[0])
    g.link(body.outputs[0], final.inputs[1])
    g.link(g.node("ShaderNodeBsdfTransparent").outputs[0], final.inputs[2])
    g.output(final.outputs[0])
    _eevee_transparency(mat)
    return mat


def _eevee_transparency(mat):
    for attr, val in (("surface_render_method", "DITHERED"), ("blend_method", "HASHED"),
                      ("use_screen_refraction", True), ("use_raytrace_refraction", True)):
        if hasattr(mat, attr):
            try:
                setattr(mat, attr, val)
            except (TypeError, ValueError):
                pass


# -- skin ---------------------------------------------------------------------
SKIN_TONES = {
    "light": (0.80, 0.58, 0.47),
    "fair": (0.72, 0.50, 0.38),
    "medium": (0.60, 0.39, 0.27),
    "tan": (0.48, 0.30, 0.19),
    "brown": (0.33, 0.20, 0.12),
    "dark": (0.20, 0.12, 0.075),
}


def skin_material(name, detail_img, head_space, eye_centres, eye_radius, brow):
    """Face skin with analytic brows and periorbital tint, in head space (metres)."""
    mat = bpy.data.materials.new(name)
    g = Graph(mat)
    tc = g.node("ShaderNodeTexCoord")
    tc.object = head_space
    p = tc.outputs["Object"]
    x, y, z = g.xyz(p)

    tone = g.rgb(SKIN_TONES["medium"], label="SkinTone")
    detail = g.image(detail_img, g.vmath("SCALE", p, scale=14.0), projection="BOX", blend=0.25)
    dr, dg, db = _split_rgb(g, detail)
    albedo = g.vmath("SCALE", tone, scale=g.add(0.9, g.mul(dr, 0.2)))

    # periorbital tint: pinker and slightly darker towards the lid margins
    dl = g.vmath("DISTANCE", p, tuple(eye_centres["L"]))
    dr_ = g.vmath("DISTANCE", p, tuple(eye_centres["R"]))
    d_eye = g.math("MINIMUM", dl, dr_)
    near = g.remap(d_eye, eye_radius + 0.007, eye_radius + 0.0012, smooth=True)
    flush = g.vmath("MULTIPLY", albedo, (1.02, 0.80, 0.78))
    albedo = g.mix(g.mul(near, 0.45), albedo, flush)

    # brows: an arched band above each eye, mirrored on |x|
    ax = g.math("ABSOLUTE", x)
    a, b, c = brow["poly"]
    centre_z = g.add(g.add(a, g.mul(ax, b)), g.mul(g.mul(ax, ax), c))
    half = g.sub(brow["half_inner"], g.mul(g.sub(ax, brow["x_inner"]), brow["taper"]))
    band = g.remap(g.math("ABSOLUTE", g.sub(z, centre_z)), half, g.mul(half, 0.45), smooth=True)
    span = g.mul(g.remap(ax, brow["x_inner"] - 0.003, brow["x_inner"] + 0.004, smooth=True),
                 g.remap(ax, brow["x_outer"] + 0.002, brow["x_outer"] - 0.008, smooth=True))
    front = g.remap(y, 0.012, -0.002, smooth=True)
    # hairs: thin bands running along the brow, tilted up at the inner end, broken by noise
    tilt = g.mul(g.sub(brow["x_inner"] + 0.012, ax), 0.9)
    along = g.add(g.sub(z, centre_z), g.mul(g.math("MAXIMUM", tilt, 0.0), 0.35))
    waves = g.node("ShaderNodeTexWave", wave_type="BANDS", bands_direction="Y")
    waves.inputs["Scale"].default_value = 1.0
    waves.inputs["Distortion"].default_value = 2.5
    waves.inputs["Detail"].default_value = 2.0
    waves.inputs["Detail Scale"].default_value = 1.5
    g.link(g.vec(g.mul(ax, 9.0), g.mul(along, 2600.0), g.mul(y, 400.0)), waves.inputs["Vector"])
    breakup = g.node("ShaderNodeTexNoise")
    breakup.inputs["Scale"].default_value = 1.0
    breakup.inputs["Detail"].default_value = 4.0
    g.link(g.vec(g.mul(ax, 900.0), g.mul(z, 2200.0), g.mul(y, 900.0)), breakup.inputs["Vector"])
    strand = g.remap(waves.outputs["Fac"], 0.35, 0.75, smooth=True)
    strand = g.mul(strand, g.remap(breakup.outputs["Fac"], 0.35, 0.6, smooth=True))
    brow_mask = g.mul(g.mul(g.mul(band, span), front), g.add(0.45, g.mul(strand, 0.55)))
    brow_col = g.rgb(brow["colour"], label="BrowColour")
    albedo = g.mix(g.mul(brow_mask, brow["density"]), albedo, brow_col)

    rough = g.add(0.42, g.mul(db, 0.16))
    rough = g.mix(near, rough, 0.30, kind="FLOAT")
    rough = g.mix(brow_mask, rough, 0.7, kind="FLOAT")

    bump = g.node("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.18
    bump.inputs["Distance"].default_value = 0.00025
    g.link(g.sub(1.0, dg), bump.inputs["Height"])

    bsdf = g.node("ShaderNodeBsdfPrincipled")
    g.link(albedo, socket(bsdf, "Base Color"))
    g.link(rough, socket(bsdf, "Roughness"))
    g.link(bump.outputs[0], socket(bsdf, "Normal"))
    set_input(bsdf, ("Subsurface Weight", "Subsurface"), 0.22)
    set_input(bsdf, ("Subsurface Radius",), (1.0, 0.35, 0.18))
    set_input(bsdf, ("Subsurface Scale",), 0.0025)
    set_input(bsdf, ("Specular IOR Level", "Specular"), 0.42)
    g.output(bsdf.outputs[0])
    return mat


def _split_rgb(g, colour):
    n = g.node("ShaderNodeSeparateColor") if hasattr(bpy.types, "ShaderNodeSeparateColor") \
        else g.node("ShaderNodeSeparateRGB")
    g.link(colour, n.inputs[0])
    return n.outputs[0], n.outputs[1], n.outputs[2]


def flat_material(name, colour, roughness=0.5, emission=None):
    mat = bpy.data.materials.new(name)
    g = Graph(mat)
    bsdf = g.node("ShaderNodeBsdfPrincipled")
    set_input(bsdf, ("Base Color",), (*colour, 1.0))
    set_input(bsdf, ("Roughness",), roughness)
    if emission is not None:
        set_input(bsdf, ("Emission Color", "Emission"), (*emission, 1.0))
        set_input(bsdf, ("Emission Strength",), 4.0)
    g.output(bsdf.outputs[0])
    return mat


def glass_material(name, tint=(0.95, 0.97, 1.0)):
    mat = bpy.data.materials.new(name)
    g = Graph(mat)
    glass = g.node("ShaderNodeBsdfGlass")
    glass.inputs["Color"].default_value = (*tint, 1.0)
    glass.inputs["IOR"].default_value = 1.5
    glass.inputs["Roughness"].default_value = 0.0
    g.output(glass.outputs[0])
    _eevee_transparency(mat)
    return mat
