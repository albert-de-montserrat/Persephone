# include("Mixer.jl")
include("Plotting/misc.jl")

function resample_fields(M, xg, yg, ipx, ipz, vx1, vy1, U)
    # Resample eigenvectors and velocity onto structured grid
    vx1g = togrid(xg, yg, vec(ipx), vec(ipz), vec(vx1))
    # vx2g = togrid(xg,yg,vec(ipx),vec(ipz),vec(vx2))
    vz1g = togrid(xg, yg, vec(ipx), vec(ipz), vec(vy1))
    # vy2g = togrid(xg,yg,vec(ipx),vec(ipz),vec(vy2))

    Ux = togrid(xg, yg, M.θ, M.r, U.x)
    Uz = togrid(xg, yg, M.θ, M.r, U.z)

    # normalise vector fields
    v1mod = @. sqrt(vx1g^2 + vz1g^2)
    # v2mod = @. sqrt(vx2g^2 + vy2g^2)
    Umod = @. sqrt(Ux^2 + Uz^2)
    vx1g ./= v1mod
    # vx2g ./ v2mod
    vz1g ./= v1mod
    # vy2g ./ v2mod
    Ux ./= Umod
    Uz ./= Umod

    # Averagea and stitch values at θ = 0 = 2π
    vx1g[:, 1] = vx1g[:, end] = mean([vx1g[:, 1], vx1g[:, end]])
    vz1g[:, 1] = vz1g[:, end] = mean([vx1g[:, 1], vx1g[:, end]])
    Ux[:, 1] = Ux[:, end] = mean([vx1g[:, 1], vx1g[:, end]])
    Uz[:, 1] = Uz[:, end] = mean([vx1g[:, 1], vx1g[:, end]])

    return vx1g, vz1g, Ux, Uz
end

#=========================================================================
GET DEM STRUCTURE:    
=========================================================================#
fname = "/home/albert/Documents/JuM2TRI/JuM2TRI/JuliaFEM/DEM/DEM_1e-3_vol20.h5"
D2 = getDEM(fname)

# Correct θ of integration points
idx = ipx .> 2π
ipx[idx] .-= 2π

# Grid to resample
xg = range(0, 2π; length=45)
yg = range(1.12, 2.12; length=10)
X = xg' .* ones(length(yg))
Y = yg .* ones(1, length(xg))
xc, yc = polar2cartesian(X, Y)

## Load file ==========================================================================
# Isotropic ---------------------------------------------------------------------------
fname = "/home/albert/Desktop/OUTPUT/Persephone/Isotropic_Tdep1e6/file_389.h5"
M, V_1e6, U_1e6, 𝓒, _, _, _, _, _, _, vx1_1e6, vx2_1e6, vy1_1e6, vy2_1e6, _ = loader(
    fname, false, false
);
λ1x_1e6, λ1z_1e6, Ux_1e6, Uz_1e6 = resample_fields(
    M, xg, yg, ipx, ipz, vx1_1e6, vy1_1e6, U_1e6
)

fname = "/home/albert/Desktop/OUTPUT/Persephone/Isotropic_Tdep1e5/file_489.h5"
M, V_1e5, U_1e5, 𝓒, _, _, _, _, _, _, vx1_1e5, vx2_1e5, vy1_1e5, vy2_1e5, _ = loader(
    fname, false, false
);
λ1x_1e5, λ1z_1e5, Ux_1e5, Uz_1e5 = resample_fields(
    M, xg, yg, ipx, ipz, vx1_1e5, vy1_1e5, U_1e5
)

fname = "/home/albert/Desktop/OUTPUT/Persephone/Isotropic_Tdep1e4/file_483.h5"
M, V_1e4, U_1e4, 𝓒, _, _, _, _, _, _, vx1_1e4, vx2_1e4, vy1_1e4, vy2_1e4, _ = loader(
    fname, false, false
);
λ1x_1e4, λ1z_1e4, Ux_1e4, Uz_1e4 = resample_fields(
    M, xg, yg, ipx, ipz, vx1_1e4, vy1_1e4, U_1e4
)

# Anisotropic ---------------------------------------------------------------------------
fname = "/home/albert/Desktop/OUTPUT/Persephone/Anisotropic_Tdep1e6/file_229.h5"
M, V_1e6_anis, U_1e6, 𝓒, _, _, _, _, _, _, vx1_1e6, vx2_1e6, vy1_1e6, vy2_1e6, _ = loader(
    fname, false, false
);
λ1x_1e6_anis, λ1z_1e6_anis, Ux_1e6_anis, Uz_1e6_anis = resample_fields(
    M, xg, yg, ipx, ipz, vx1_1e6, vy1_1e6, U_1e6
)

fname = "/home/albert/Desktop/OUTPUT/Persephone/Anisotropic_Tdep1e5/file_195.h5"
M, V_1e5_anis, U_1e5, 𝓒, _, _, _, _, _, _, vx1_1e5, vx2_1e5, vy1_1e5, vy2_1e5, _ = loader(
    fname, false, false
);
λ1x_1e5_anis, λ1z_1e5_anis, Ux_1e5_anis, Uz_1e5_anis = resample_fields(
    M, xg, yg, ipx, ipz, vx1_1e5, vy1_1e5, U_1e5
)

## Calculate viscosity
# η = getviscosity(Viso.T, "TemperatureDependantIsotropic")

## Plot
xz = [M.x M.z]
x, z = polar2cartesian(ipx, ipz)

scaling = 1.25e1
scaling1 = 1.25e1

# Plot -> T Isotropic ================================================================
## Ra = 1e6 ---------------------------------------------------------------------------
fig = Figure(; resolution=(600, 1500))

ax11 = fig[1, 1] = Axis(fig)
m1 = mesh!(ax11, xz, M.e2n; color=V_1e6.T, colormap=ColorSchemes.vik, shading=false)

ax12 = fig[2, 1] = Axis(fig)
q1 = arrows!(
    ax12,
    vec(xc),
    vec(yc),
    0.5 * vec(λ1x_1e6) / scaling,
    0.5 * vec(λ1z_1e6) / scaling;
    arrowsize=0,
    arrowcolor=:red,
    linecolor=:red,
    linewidth=2,
)
q2 = arrows!(
    ax12,
    vec(xc),
    vec(yc),
    -0.5 * vec(λ1x_1e6) / scaling,
    -0.5 * vec(λ1z_1e6) / scaling;
    arrowsize=0,
    arrowcolor=:red,
    linecolor=:red,
    linewidth=2,
)
q3 = arrows!(
    ax12,
    vec(xc),
    vec(yc),
    vec(Ux_1e6) / scaling1,
    vec(Uz_1e6) / scaling1;
    arrowsize=2e-2,
    arrowcolor=:black,
    linecolor=:black,
    linewidth=2,
)

xlims!(ax11, (-2.12, 0))
ylims!(ax11, (0, 2.12))

xlims!(ax12, (-2.12, 0))
ylims!(ax12, (-2.12, 0))

hidedecorations!(ax12; ticklabels=false)
hidedecorations!(ax11; ticklabels=false)

hidexdecorations!(ax11)
hidexdecorations!(ax12)

ax11.yticks = [0.0, 1, 2.0]
ax12.yticks = [0.0, -1, -2.0]

ax11.aspect = DataAspect()
ax12.aspect = DataAspect()

# cbarT = Colorbar(fig[1, 3], m1, label = "T", width = 25, fontsize = 6)
# cbarT.tellheight = true

## Ra = 1e5 ---------------------------------------------------------------------------
ax21 = fig[3, 1] = Axis(fig)
m1 = mesh!(ax21, xz, M.e2n; color=V_1e5.T, colormap=ColorSchemes.vik, shading=false)

ax22 = fig[4, 1] = Axis(fig)
q1 = arrows!(
    ax22,
    vec(xc),
    vec(yc),
    0.5 * vec(λ1x_1e5) / scaling,
    0.5 * vec(λ1z_1e5) / scaling;
    arrowsize=0,
    arrowcolor=:red,
    linecolor=:red,
    linewidth=2,
)
q2 = arrows!(
    ax22,
    vec(xc),
    vec(yc),
    -0.5 * vec(λ1x_1e5) / scaling,
    -0.5 * vec(λ1z_1e5) / scaling;
    arrowsize=0,
    arrowcolor=:red,
    linecolor=:red,
    linewidth=2,
)
q3 = arrows!(
    ax22,
    vec(xc),
    vec(yc),
    vec(Ux_1e5) / scaling1,
    vec(Uz_1e5) / scaling1;
    arrowsize=2e-2,
    arrowcolor=:black,
    linecolor=:black,
    linewidth=2,
)

xlims!(ax22, (-2.12, 0))
ylims!(ax22, (-2.12, 0))

xlims!(ax21, (-2.12, 0))
ylims!(ax21, (0, 2.12))

ax21.yticks = [0.0, 1, 2.0]
ax22.yticks = [0.0, -1, -2.0]
ax22.xticks = [-2.0, -1, 0]

hidedecorations!(ax22; ticklabels=false)
hidedecorations!(ax21; ticklabels=false)

hidexdecorations!(ax21)
# hidexdecorations!(ax22)

ax21.aspect = DataAspect()
ax22.aspect = DataAspect()

# Plot -> T Anisotropic ================================================================
## Ra = 1e6 ---------------------------------------------------------------------------
ax31 = fig[1, 2] = Axis(fig)
m1 = mesh!(ax31, xz, M.e2n; color=V_1e6_anis.T, colormap=ColorSchemes.vik, shading=false)

ax32 = fig[2, 2] = Axis(fig)
q1 = arrows!(
    ax32,
    vec(xc),
    vec(yc),
    0.5 * vec(λ1x_1e6_anis) / scaling,
    0.5 * vec(λ1z_1e6_anis) / scaling;
    arrowsize=0,
    arrowcolor=:red,
    linecolor=:red,
    linewidth=2,
)
q2 = arrows!(
    ax32,
    vec(xc),
    vec(yc),
    -0.5 * vec(λ1x_1e6_anis) / scaling,
    -0.5 * vec(λ1z_1e6_anis) / scaling;
    arrowsize=0,
    arrowcolor=:red,
    linecolor=:red,
    linewidth=2,
)
q3 = arrows!(
    ax32,
    vec(xc),
    vec(yc),
    vec(Ux_1e6_anis) / scaling1,
    vec(Uz_1e6_anis) / scaling1;
    arrowsize=2e-2,
    arrowcolor=:black,
    linecolor=:black,
    linewidth=2,
)

xlims!(ax31, (0, 2.12))
ylims!(ax31, (0, 2.12))

xlims!(ax32, (0, 2.12))
ylims!(ax32, (-2.12, 0))

hidedecorations!(ax32)
hidedecorations!(ax31)

ax31.aspect = DataAspect()
ax32.aspect = DataAspect()

## Ra = 1e5 ---------------------------------------------------------------------------
ax41 = fig[3, 2] = Axis(fig)
m1 = mesh!(ax41, xz, M.e2n; color=V_1e5_anis.T, colormap=ColorSchemes.vik, shading=false)

ax42 = fig[4, 2] = Axis(fig)
q1 = arrows!(
    ax42,
    vec(xc),
    vec(yc),
    0.5 * vec(λ1x_1e5_anis) / scaling,
    0.5 * vec(λ1z_1e5_anis) / scaling;
    arrowsize=0,
    arrowcolor=:red,
    linecolor=:red,
    linewidth=2,
)
q2 = arrows!(
    ax42,
    vec(xc),
    vec(yc),
    -0.5 * vec(λ1x_1e5_anis) / scaling,
    -0.5 * vec(λ1z_1e5_anis) / scaling;
    arrowsize=0,
    arrowcolor=:red,
    linecolor=:red,
    linewidth=2,
)
q3 = arrows!(
    ax42,
    vec(xc),
    vec(yc),
    vec(Ux_1e5_anis) / scaling1,
    vec(Uz_1e5_anis) / scaling1;
    arrowsize=2e-2,
    arrowcolor=:black,
    linecolor=:black,
    linewidth=2,
)

xlims!(ax42, (0, 2.12))
ylims!(ax42, (-2.12, 0))

xlims!(ax41, (0, 2.12))
ylims!(ax41, (0, 2.12))

ax42.yticks = [0.0, -1, -2.0]
ax42.xticks = [0, 1, 2]

hidedecorations!(ax41)
hidedecorations!(ax42; ticklabels=false)

hideydecorations!(ax42)

ax41.aspect = DataAspect()
ax42.aspect = DataAspect()

label_a = fig[1, 1:2, TopLeft()] = Label(fig, "Ra = 1e6"; textsize=18, halign=:right)
label_a.padding = (0, -300, 10, 0)

label_b = fig[3, 1:2, TopLeft()] = Label(fig, "Ra = 1e5"; textsize=18, halign=:right)
label_b.padding = (0, -300, 10, 0)

trim!(fig.layout)

fig

save("Temperate_FSE_Tdep.png", fig)

# ## Ra = 1e4 ---------------------------------------------------------------------------
# ax31 = fig[5, 1] = Axis(fig)
# m1 = mesh!(ax31, xz, M.e2n, color = T3, colormap = ColorSchemes.vik, shading = false) 

# ax32 = fig[6, 1] = Axis(fig)
# q1 = arrows!(ax32, vec(xc), vec(yc), 
#     0.5*vec(λ1x_1e4)/scaling, 0.5*vec(λ1z_1e4)/scaling,
#     arrowsize = 0, arrowcolor = :red, linecolor = :red, linewidth =3)
# q2 = arrows!(ax32, vec(xc), vec(yc), 
#     -0.5*vec(λ1x_1e4)/scaling, -0.5*vec(λ1z_1e4)/scaling,
#     arrowsize = 0, arrowcolor = :red, linecolor = :red, linewidth =3)
# q3 = arrows!(ax32, vec(xc), vec(yc), 
#     vec(Ux_1e4)/scaling1, vec(Uz_1e4)/scaling1,
#     arrowsize = 2e-2, arrowcolor = :black, linecolor = :black, linewidth =3)

# xlims!(ax32,(-2.12, 0))
# ylims!(ax32,(-2.12, 0))

# xlims!(ax31,(-2.12, 0))
# ylims!(ax31,(0, 2.12))

# ax31.yticks = [0.0, 1, 2.0]
# ax32.yticks = [0.0, -1, -2.0]
# ax32.xticks = [-2.0, -1, 0]

# hidedecorations!(ax32,ticklabels=false)
# hidedecorations!(ax31,ticklabels=false)

# hidexdecorations!(ax31)

# ax31.aspect = DataAspect()
# ax32.aspect = DataAspect()

# fig

# record(fig, "output.mp4", 1:length(T)) do i
#     m1[:color] = T[i]

#     # puv = [Vec2f0(Fx1[i][j], Fy1[i][j]) for j in eachindex(Fy1[i])]
#     # q1[:directions].val = puv

#     # puv = [Vec2f0(-Fx1[i][j], -Fy1[i][j]) for j in eachindex(Fy1[i])]
#     # q2[:directions].val = puv

#     # puv = [Vec2f0(Ux[i][j], Uz[i][j]) for j in eachindex(Fy1[i])]
#     # q3[:directions].val = puv

#     # title[] = "t = $(round(t[i]; sigdigits = 4))"
#     sleep(1)
# end

# # ###-----------------------------
# # # Plot -> T anisotropy
# # ax2 = fig[1, 2] = Axis(fig, title = "Isotropic")
# # m2 = mesh!(ax2, xz, M.e2n,color = V.T, colormap = ColorSchemes.thermal,shading = false) 
# # arrows!(ax2,vec(x),vec(z), 0.5*vec(vx1)/scaling, 0.5*vec(vy1)/scaling, arrowsize = 1e-5)
# # xlims!(ax2,(0,2.12))
# # ylims!(ax2,(0,2.12))
# # ax2.aspect = DataAspect()
# # # cbarT = Colorbar(fig[1, 2], m1, label = "T", width = 25, fontsize = 6)
# # linkaxes!(ax1,ax2)
# # xlims!(ax2,(0,2.12))
# # ylims!(ax2,(0,2.12))

# # # Plot -> T isotropy
# # ax2 = fig[1, 3] = Axis(fig, title = "Isotropic")
# # m2 = mesh!(ax2, xz, M.e2n, color = Viso.T, colormap = ColorSchemes.thermal, shading = false) 
# # arrows!(ax2,vec(xc),vec(yc), 0.5*vec(vx1_iso)/scaling, 0.5vec(vy1_iso)/scaling, arrowsize = 1e-5)
# # arrows!(ax2,vec(xc),vec(yc), -0.5*vec(vx1_iso)/scaling, -0.5vec(vy1_iso)/scaling, arrowsize = 1e-5)

# # # Plot -> η anisotropy
# # ax3 = fig[2, 2] = Axis(fig)
# # m3 = mesh!(ax3, xz,M.e2n,color = log10.(η.val), colormap = ColorSchemes.vik, shading = false) 
# # m3.colorrange = extrema(log10.(ηiso.val))

# # # Plot -> η isotropy
# # ax4 = fig[2, 3] = Axis(fig)
# # m4 = mesh!(ax4, xz,M.e2n,color = log10.(ηiso.val), colormap = ColorSchemes.vik, shading = false) 
# # m4.colorrange = extrema(log10.(ηiso.val))

# # linkaxes!(ax1,ax2)
# # linkaxes!(ax3,ax4)

# # xlims!(ax2,(0,2.12))
# # ylims!(ax2,(0,2.12))
# # xlims!(ax3,(0,2.12))
# # ylims!(ax3,(-2.12,0))
# # xlims!(ax5,(-2.12,0))
# # ylims!(ax5,(0,2.12))
# # xlims!(ax6,(-2.12,0))
# # ylims!(ax6,(-2.12,0))

# # hidedecorations!(ax1,ticklabels=false)
# # hidedecorations!(ax2,ticklabels=false)
# # hidedecorations!(ax3,ticklabels=false)
# # hidedecorations!(ax4,ticklabels=false)
# # hidedecorations!(ax5,ticklabels=false)
# # hidedecorations!(ax6,ticklabels=false)

# # hideydecorations!(ax1)
# # hideydecorations!(ax2)
# # hideydecorations!(ax3)
# # hideydecorations!(ax4)
# # hidexdecorations!(ax1)
# # hidexdecorations!(ax2)
# # hidexdecorations!(ax5)

# # ax1.aspect = DataAspect()
# # ax2.aspect = DataAspect()
# # ax3.aspect = DataAspect()
# # ax4.aspect = DataAspect()
# # ax5.aspect = DataAspect()
# # ax6.aspect = DataAspect()

# # cbarT = Colorbar(fig[1, 4], m1, label = "T", width = 25, fontsize = 6)
# # cbarUθ = Colorbar(fig[2, 4], m3, label = "Tangential Velocity", width = 25, fontsize = 6)

# # cbarRadial = Colorbar(fig[1, 5], m5, label = "Radial anisotropy", width = 25, fontsize = 6)
# # m5.colorrange = (1.0, 2.5)

# # cbar_n55 = Colorbar(fig[2, 5], m6, label = "Azimuthal anisotropy", width = 25)
# # m6.colorrange = (0, 0.015)

# # trim!(fig.layout)
# # fig

# # xy = [Point2D{Cartesian}(ipx[i], ipz[i]) for i in eachindex(ipx)]
# # Uxg = togrid(xg, yg, xy, vx1)

# # X = xg'.*ones(length(yg))
# # Y = yg.*ones(1,length(xg))

# # Uxg, Uzg, xg, zg = structured_velocity(M.θ, M.r, Uiso)

# # strf = streamline(Uxg', Uzg', xg, zg )
# # xc, zc = polar2cartesian(xg,zg)
# # contour(xg[1,:],zg[:,1],strf,levels=20)

# x = [1,2]
# y = [2,2]
# u = rand(2)
# v = rand(2)
# a = arrows(x,y,u,v,arrowsize=0)

# puv = [Vec2f0(rand(), rand()) for j in 1:length(u)]
# a.plot[2].val = puv
# a.plot[:directions] = puv
