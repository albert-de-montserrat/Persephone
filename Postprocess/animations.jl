include("Mixer.jl")
include(joinpath(pwd(), "Postprocess/plot_fields.jl"))

path = "/home/albert/Documents/JuM2TRI/NoAvx/output/"

fanimation = string(path, "Anisotropic_1e4_fulltensor3")

nr = Int(1+32)
nθ = Int(256)
gr = Grid(nθ, nr)

animation_node_black(gr, fanimation)


du -h --max-depth=1 | sort -rh