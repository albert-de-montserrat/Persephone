include(joinpath(pwd(), "Postprocess/plot_fields.jl"))

nr = Int(1+32)
nθ = Int(256)
gr = Grid(nθ, nr)

path = "/home/albert/Documents/AnnulusBenchmarksCluster"
fpath = joinpath(path, "Rand_Tdep_Anisotropic_1e5")
fnumber = 25
file = string("file_", fnumber, ".h5")
fname = joinpath(fpath, file)
fstats = joinpath(fpath, "statistics.txt")

# plot_stats(fstats)
            
# plot_anisotropy(fname, gr, isave = "no")

M, V, U, 𝓒, Fxx, Fzz, Fxz, Fzx, a1, a2, vx1, vx2, vy1, vy2 = loader(fname, isotropic=true);

# Plot
figT = plot_node(V.T, gr)

# save("Temperature_1e4.png", figT)

# Grid to resample
Ael = element_area(gr)

η11 = ip2node(gr.e2n, Ael, 𝓒.η11[:, 1:6])
η33 = ip2node(gr.e2n, Ael, 𝓒.η33[:, 1:6])
η55 = ip2node(gr.e2n, Ael, 𝓒.η55[:, 1:6])
η13 = ip2node(gr.e2n, Ael, 𝓒.η13[:, 1:6])
η15 = ip2node(gr.e2n, Ael, 𝓒.η15[:, 1:6])
η35 = ip2node(gr.e2n, Ael, 𝓒.η35[:, 1:6])

figT = plot_node(η11, gr)
figT = plot_node(η33, gr)
figT = plot_node(η55, gr)
figT = plot_node(η13, gr)
figT = plot_node(η15, gr)
figT = plot_node(η35, gr)
figT = plot_node(η35-η15, gr)

figT = plot_node(U[1:2:end], gr)

a1 = [FSE[i].a1 for i in CartesianIndices(FSE)]
a1n = ip2node(gr.e2n, Ael, a2[:, 1:6])
figT = plot_node(a1n, gr)

# Grid to resample
xg = range(0, 2π, length = 90)
yg = range(1.22, 2.22, length = 25)
X = xg'.*ones(length(yg))
Y = yg.*ones(1,length(xg))
xc, yc = polar2cartesian(X,Y)

F = [@SMatrix [Fxx[i] Fxz[i]; Fzx[i] Fzz[i]] for i in CartesianIndices(Fxx)]
FSE = computeFSE(F)

Fx1 = [FSE[i].x1 for i in CartesianIndices(FSE)]
Fy1 = [FSE[i].y1 for i in CartesianIndices(FSE)]

α = atand.(Fy1,Fx1)
αn = ip2node(gr.e2n, Ael, α)
figT = plot_node(αn, gr)

# Resample eigenvectors and velocity onto structured grid
vx1g = togrid(xg, yg, vec(ipx), vec(ipz), vec(vx1))
vy1g = togrid(xg, yg, vec(ipx), vec(ipz), vec(vy1))

scaling = 5e0
arrows!(vec(xc), vec(yc), 
        vec(vx1g)/scaling, vec(vy1g)/scaling,
        arrowsize = 1e-2, arrowcolor = :black, linecolor = :green, linewidth =2)
        
scaling = 1e3
arrows(gr.x, gr.z,
        U.x/scaling, U.z/scaling,
        arrowsize = 5e-3, arrowcolor = :red, linecolor = :black, linewidth =1)
