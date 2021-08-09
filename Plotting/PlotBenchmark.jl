# include("Mixer.jl")
include("Plotting/misc.jl")

#=========================================================================
GET DEM STRUCTURE:    
=========================================================================#
fname = "/home/albert/Documents/JuM2TRI/JuM2TRI/JuliaFEM/DEM/DEM_1e-3_vol20.h5"
D2 = getDEM(fname)

idx = ipx .> 2π
ipx[idx] .-= 2π

path = "/home/albert/Documents/JuM2TRI/JuM2TRI/JuliaFEM/output/IsoviscousIsotropic_1e4_full_tensor_2/"

path = "/home/albert/Desktop/Persephone2Cluster/ClusterOutput/Isotropic/Tdep/LinearT/1e4/"
# Grid to resample
xg = range(0, 2π; length=90)
yg = range(1.12, 2.12; length=10)
X = xg' .* ones(length(yg))
Y = yg .* ones(1, length(xg))
xc, yc = polar2cartesian(X, Y)

files = readdir(path)
nfiles = 1:1:length(readdir(path))
Fx1 = [fill(0.0, size(X, 1), size(X, 2)) for _ in nfiles]
Fy1 = [fill(0.0, size(X, 1), size(X, 2)) for _ in nfiles]
Ux = [fill(0.0, size(X, 1), size(X, 2)) for _ in nfiles]
Uz = [fill(0.0, size(X, 1), size(X, 2)) for _ in nfiles]
Uθ = Vector{Float64}[]
Ur = Vector{Float64}[]
T = Vector{Float64}[]
Uxx = Vector{Float64}[]
Uzz = Vector{Float64}[]
t = Vector{Float64}(undef, length(nfiles))
M = 0
Time = 0.0

# for (i, fnumber) in enumerate(nfiles)
for (i, file) in enumerate(files)
    # file = string("file_",fnumber,".h5")
    fname = joinpath(path, file)
    # Load file
    M, V, U, 𝓒, _, _, _, _, _, _, vx1, vx2, vy1, vy2, Time = loader(fname, false, true)

    # x, z = M.x, M.z
    # θ, r = M.θ, M.r

    ## Remove mantle wind
    # U.θ .= remove_wind(M.r,U.θ)

    # Resample eigenvectors and velocity onto structured grid
    vx1g = togrid(xg, yg, vec(ipx), vec(ipz), vec(vx1))
    # vx2g = togrid(xg,yg,vec(ipx),vec(ipz),vec(vx2))
    vy1g = togrid(xg, yg, vec(ipx), vec(ipz), vec(vy1))
    # vy2g = togrid(xg,yg,vec(ipx),vec(ipz),vec(vy2))

    ux = togrid(xg, yg, M.θ, M.r, U.x)
    uz = togrid(xg, yg, M.θ, M.r, U.z)

    # normalise vector fields
    v1mod = @. sqrt(vx1g^2 + vy1g^2)
    # v2mod = @. sqrt(vx2g^2 + vy2g^2)
    Umod = @. sqrt(ux^2 + uz^2)
    Fx1[i] .= vx1g ./ v1mod
    # vx2g ./ v2mod
    Fy1[i] .= vy1g ./ v1mod
    # vy2g ./ v2mod
    Ux[i] .= ux ./ Umod
    Uz[i] .= uz ./ Umod

    push!(T, V.T)
    push!(Uxx, U.x)
    push!(Uzz, U.z)

    t[i] = Time
end

## Calculate viscosity
# η = getviscosity(Viso.T, "TemperatureDependantIsotropic")

## Plot
xz = [M.x M.z]
x, z = polar2cartesian(ipx, ipz)
xc, yc = polar2cartesian(X, Y)

scaling = 1.25e1
scaling1 = 1.25e1

# Plotting fields 
nn = length(nfiles)
make_plot(nn, T, Fx1, Fy1, Ux, Uz, Uxx, Uzz, t, xz, M)

function make_plot(nn, T, Fx1, Fy1, Ux, Uz, Uxx, Uzz, t, xz, M)
    Tplot = T[nn]
    Fx1_plot = Fx1[nn]
    Fy1_plot = Fy1[nn]
    Ux_plot = Ux[nn]
    Uz_plot = Uz[nn]
    Vx = Uxx[nn]
    Vz = Uzz[nn]
    Time = t[nn]

    # Plot -> T anisotropy
    fig = Figure(; resolution=(1200, 900), backgroundcolor=RGBf0(0.98, 0.98, 0.98))

    ax0 = fig[1, 1] = Axis(fig; title="t = $Time")
    # q1 = arrows!(ax0, vec(xc), vec(yc), 
    #     0.5*vec(Fx1_plot)/scaling, 0.5*vec(Fy1_plot)/scaling,
    #     arrowsize = 0, arrowcolor = :red, linecolor = :red, linewidth =3)
    # q2 = arrows!(ax0, vec(xc), vec(yc), 
    #     -0.5*vec(Fx1_plot)/scaling, -0.5*vec(Fy1_plot)/scaling,
    #     arrowsize = 0, arrowcolor = :red, linecolor = :red, linewidth =3)
    q3 = arrows!(
        ax0,
        vec(xc),
        vec(yc),
        vec(Ux_plot) / scaling1,
        vec(Uz_plot) / scaling1;
        arrowsize=3e-2,
        arrowcolor=:black,
        linecolor=:black,
        linewidth=2,
    )

    ax1 = fig[1, 2] = Axis(fig; title="t = $Time")
    # m1 = mesh!(ax1, xz, M.e2n, color = Tplot, colormap = ColorSchemes.vik, shading = false) 
    # m1 = mesh!(ax1, xz[1:nT,:], M.e2n, color = U.θ[1:nT], colormap = ColorSchemes.vik, shading = false) 
    m1 = mesh!(ax1, xz, M.e2n; color=Vx, colormap=ColorSchemes.vik, shading=false)

    GLMakie.xlims!(ax0, (-2.12, 0))
    GLMakie.ylims!(ax0, (0, 2.12))
    # GLMakie.xlims!(ax1,(0, 2.12))
    # GLMakie.ylims!(ax1,(0, 2.12))

    hidedecorations!(ax0; ticklabels=false)
    hidedecorations!(ax1; ticklabels=false)
    hideydecorations!(ax1)

    ax1.aspect = DataAspect()
    ax0.aspect = DataAspect()

    cbarT = Colorbar(fig[1, 3], m1; label="T", width=25, fontsize=6)
    cbarT.tellheight = true
    return fig

    # record(fig, "output.mp4", 1:length(T)) do i
    #     m1[:color] = T[i]
    #     sleep(1)
    # end

end

record(fig, "output.mp4", 1:length(T)) do i
    m1[:color] = T[i]

    # puv = [Vec2f0(Fx1[i][j], Fy1[i][j]) for j in eachindex(Fy1[i])]
    # q1[:directions].val = puv

    # puv = [Vec2f0(-Fx1[i][j], -Fy1[i][j]) for j in eachindex(Fy1[i])]
    # q2[:directions].val = puv

    # puv = [Vec2f0(Ux[i][j], Uz[i][j]) for j in eachindex(Fy1[i])]
    # q3[:directions].val = puv

    # title[] = "t = $(round(t[i]; sigdigits = 4))"
    sleep(1)
end

# ###-----------------------------
# # Plot -> T anisotropy
# ax2 = fig[1, 2] = Axis(fig, title = "Isotropic")
# m2 = mesh!(ax2, xz, M.e2n,color = V.T, colormap = ColorSchemes.thermal,shading = false) 
# arrows!(ax2,vec(x),vec(z), 0.5*vec(vx1)/scaling, 0.5*vec(vy1)/scaling, arrowsize = 1e-5)
# xlims!(ax2,(0,2.12))
# ylims!(ax2,(0,2.12))
# ax2.aspect = DataAspect()
# # cbarT = Colorbar(fig[1, 2], m1, label = "T", width = 25, fontsize = 6)
# linkaxes!(ax1,ax2)
# xlims!(ax2,(0,2.12))
# ylims!(ax2,(0,2.12))

# # Plot -> T isotropy
# ax2 = fig[1, 3] = Axis(fig, title = "Isotropic")
# m2 = mesh!(ax2, xz, M.e2n, color = Viso.T, colormap = ColorSchemes.thermal, shading = false) 
# arrows!(ax2,vec(xc),vec(yc), 0.5*vec(vx1_iso)/scaling, 0.5vec(vy1_iso)/scaling, arrowsize = 1e-5)
# arrows!(ax2,vec(xc),vec(yc), -0.5*vec(vx1_iso)/scaling, -0.5vec(vy1_iso)/scaling, arrowsize = 1e-5)

# # Plot -> η anisotropy
# ax3 = fig[2, 2] = Axis(fig)
# m3 = mesh!(ax3, xz,M.e2n,color = log10.(η.val), colormap = ColorSchemes.vik, shading = false) 
# m3.colorrange = extrema(log10.(ηiso.val))

# # Plot -> η isotropy
# ax4 = fig[2, 3] = Axis(fig)
# m4 = mesh!(ax4, xz,M.e2n,color = log10.(ηiso.val), colormap = ColorSchemes.vik, shading = false) 
# m4.colorrange = extrema(log10.(ηiso.val))

# linkaxes!(ax1,ax2)
# linkaxes!(ax3,ax4)

# xlims!(ax2,(0,2.12))
# ylims!(ax2,(0,2.12))
# xlims!(ax3,(0,2.12))
# ylims!(ax3,(-2.12,0))
# xlims!(ax5,(-2.12,0))
# ylims!(ax5,(0,2.12))
# xlims!(ax6,(-2.12,0))
# ylims!(ax6,(-2.12,0))

# hidedecorations!(ax1,ticklabels=false)
# hidedecorations!(ax2,ticklabels=false)
# hidedecorations!(ax3,ticklabels=false)
# hidedecorations!(ax4,ticklabels=false)
# hidedecorations!(ax5,ticklabels=false)
# hidedecorations!(ax6,ticklabels=false)

# hideydecorations!(ax1)
# hideydecorations!(ax2)
# hideydecorations!(ax3)
# hideydecorations!(ax4)
# hidexdecorations!(ax1)
# hidexdecorations!(ax2)
# hidexdecorations!(ax5)

# ax1.aspect = DataAspect()
# ax2.aspect = DataAspect()
# ax3.aspect = DataAspect()
# ax4.aspect = DataAspect()
# ax5.aspect = DataAspect()
# ax6.aspect = DataAspect()

# cbarT = Colorbar(fig[1, 4], m1, label = "T", width = 25, fontsize = 6)
# cbarUθ = Colorbar(fig[2, 4], m3, label = "Tangential Velocity", width = 25, fontsize = 6)

# cbarRadial = Colorbar(fig[1, 5], m5, label = "Radial anisotropy", width = 25, fontsize = 6)
# m5.colorrange = (1.0, 2.5)

# cbar_n55 = Colorbar(fig[2, 5], m6, label = "Azimuthal anisotropy", width = 25)
# m6.colorrange = (0, 0.015)

# trim!(fig.layout)
# fig

# xy = [Point2D{Cartesian}(ipx[i], ipz[i]) for i in eachindex(ipx)]
# Uxg = togrid(xg, yg, xy, vx1)

# X = xg'.*ones(length(yg))
# Y = yg.*ones(1,length(xg))

# Uxg, Uzg, xg, zg = structured_velocity(M.θ, M.r, Uiso)

# strf = streamline(Uxg', Uzg', xg, zg )
# xc, zc = polar2cartesian(xg,zg)
# contour(xg[1,:],zg[:,1],strf,levels=20)

x = [1, 2]
y = [2, 2]
u = rand(2)
v = rand(2)
a = arrows(x, y, u, v; arrowsize=0)

puv = [Vec2f0(rand(), rand()) for j in 1:length(u)]
a.plot[2].val = puv
a.plot[:directions] = puv
