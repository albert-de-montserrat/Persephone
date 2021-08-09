# include("Mixer.jl")
include("Plotting/misc.jl")

#=========================================================================
GET DEM STRUCTURE:    
=========================================================================#
function make_plot(fname1, fname2, ipx, D)

    # Isotropic η55 
    η55_0 = maximum(D.𝓒[:, 5])

    fname = "/home/albert/Documents/JuM2TRI/JuM2TRI/JuliaFEM/output/IsoviscousIsotropic_1e4_tsearch_parallel/file_1.h5"
    M, V, U, 𝓒, Fxx, Fzz, Fxz, Fzx, a1, a2, vx1, vx2, vy1, vy2 = loader(fname)

    FSE = [
        FiniteStrainEllipsoid{Float64}(1.0, 0.0, 0.0, 1.0, a1[i], a2[i]) for
        i in CartesianIndices(vx1)
    ]

    𝓒 = anisotropic_tensor_postprocess(FSE, D)
    𝓒r = cartesian_tensor(𝓒, vx1, vx2, vy1, vy2, ipx)

    # 𝓒 = unrotate_anisotropic_tensor(𝓒,vx1,vx2,vy1,vy2)

    η11 = [𝓒[i][1, 1] for i in CartesianIndices(𝓒)]
    η22 = [𝓒[i][2, 2] for i in CartesianIndices(𝓒)]
    η12 = [𝓒[i][1, 2] for i in CartesianIndices(𝓒)]
    η45 = [𝓒[i][4, 5] for i in CartesianIndices(𝓒)]
    η44 = [𝓒[i][4, 4] for i in CartesianIndices(𝓒)]
    η55 = [𝓒[i][5, 5] for i in CartesianIndices(𝓒)]
    η66 = [𝓒[i][6, 6] for i in CartesianIndices(𝓒)]
    anisotropy = @. 1 - η55 / η55_0

    # normalise anisotropy
    anisotropy_normalised = anisotropy ./ maximum(anisotropy)

    # Radial anisotropy 
    N = @. 0.125 * (η11 + η22) - 0.25 * η12 + 0.5 * η66
    L = @. 0.5 * (η44 + η55)
    χ = N ./ L
    χ = ip2node(M.e2n', area_el, χ)

    # Azimuthal anisotropy
    Gc = @. 0.5 * (η55 - η44)
    Gs = η45
    G = @. 0.5 * (Gc^2 + Gs^2)
    G = ip2node(M.e2n', area_el, G)

    a1 = ip2node(M.e2n', area_el, a1)
    a2 = ip2node(M.e2n', area_el, a2)
    η55 = ip2node(M.e2n', area_el, η55)

    # anisotropy_normalised = ip2node(M.e2n',area_el,anisotropy)

    # η11 = ip2node(M.e2n',area_el,η11)
    # η33 = ip2node(M.e2n',area_el,η33)
    # η55 = ip2node(M.e2n',area_el,η55)
    # η13 = ip2node(M.e2n',area_el,η13)

    # fname = "/home/albert/Documents/JuM2TRI/JuM2TRI/JuliaFEM/output/Isotropic1e6_Tdep/file_120.h5"
    M, Viso, Uiso, 𝓒, _, _, _, _, _, _, vx1_iso, vx2_iso, vy1_iso, vy2_iso = loader(
        fname2, false
    )

    x, z = M.x, M.z
    θ, r = M.θ, M.r

    ## Remove mantle wind
    # U.θ .= remove_wind(M.r,U.θ)

    # Resample eigenvectors
    xg = range(0, 2π; length=45)
    yg = range(1.12, 2.12; length=10)
    X = xg' .* ones(length(yg))
    Y = yg .* ones(1, length(xg))
    xc, yc = polar2cartesian(X, Y)

    vx1 = togrid(xg, yg, vec(ipx), vec(ipz), vec(vx1))
    vx2 = togrid(xg, yg, vec(ipx), vec(ipz), vec(vx2))
    vx1_iso = togrid(xg, yg, vec(ipx), vec(ipz), vec(vx1_iso))
    vx2_iso = togrid(xg, yg, vec(ipx), vec(ipz), vec(vx2_iso))

    vy1 = togrid(xg, yg, vec(ipx), vec(ipz), vec(vy1))
    vy2 = togrid(xg, yg, vec(ipx), vec(ipz), vec(vy2))
    vy1_iso = togrid(xg, yg, vec(ipx), vec(ipz), vec(vy1_iso))
    vy2_iso = togrid(xg, yg, vec(ipx), vec(ipz), vec(vy2_iso))

    anisotropy_normalisedg = togrid(xg, yg, vec(ipx), vec(ipz), vec(anisotropy_normalised))

    # normalise vector fields
    v_anis_mod = @. sqrt(vx1^2 + vy1^2)
    v_iso_mod = @. sqrt(vx1_iso^2 + vy1_iso^2)
    vx1 ./= v_anis_mod
    vx2 ./= v_anis_mod
    vx1_iso ./= v_iso_mod
    vx2_iso ./= v_iso_mod

    ## Calculate viscosity
    η = getviscosity(V.T, "TemperatureDependantIsotropic")
    ηiso = getviscosity(Viso.T, "TemperatureDependantIsotropic")

    ## Plot
    xz = [M.x M.z]
    x, z = polar2cartesian(ipx, ipz)
    nT = 64000
    scaling = 1e1

    fig = Figure(; resolution=(1700, 900), backgroundcolor=RGBf0(0.98, 0.98, 0.98))

    # Plot -> Radial anisotropy
    ax5 = fig[1, 1] = Axis(fig)
    m5 = mesh!(ax5, xz, M.e2n; color=χ, colormap=ColorSchemes.Dark2_8, shading=false)

    # Plot -> Azimuthal anisotropy
    ax6 = fig[2, 1] = Axis(fig)
    m6 = mesh!(ax6, xz, M.e2n; color=G, colormap=ColorSchemes.Paired_12, shading=false)

    # Plot -> T anisotropy
    ax1 = fig[1, 2] = Axis(fig; title="Anisotropic")
    m1 = mesh!(ax1, xz, M.e2n; color=V.T, colormap=ColorSchemes.thermal, shading=false)
    arrows!(
        ax1,
        vec(xc),
        vec(yc),
        0.5 * vec(vx1) / scaling,
        0.5 * vec(vy1) / scaling;
        arrowsize=0,
        linecolor=:red,
        linewidth=3,
    )
    arrows!(
        ax1,
        vec(xc),
        vec(yc),
        -0.5 * vec(vx1) / scaling,
        -0.5 * vec(vy1) / scaling;
        arrowsize=0,
        linecolor=:red,
        linewidth=3,
    )

    # Plot -> T isotropy
    ax2 = fig[1, 1] = Axis(fig; title="Isotropic")
    m2 = mesh!(
        ax2,
        xz[1:nT, :],
        M.e2n;
        color=Viso.T[1:nT],
        colormap=ColorSchemes.thermal,
        shading=false,
    )
    arrows!(
        ax2,
        vec(xc),
        vec(yc),
        0.5 * vec(vx1_iso) / scaling,
        0.5vec(vy1_iso) / scaling;
        arrowsize=0,
        linecolor=:red,
        linewidth=3,
    )
    arrows!(
        ax2,
        vec(xc),
        vec(yc),
        -0.5 * vec(vx1_iso) / scaling,
        -0.5vec(vy1_iso) / scaling;
        arrowsize=0,
        linecolor=:red,
        linewidth=3,
    )

    # Plot -> η anisotropy
    ax3 = fig[2, 2] = Axis(fig)
    m3 = mesh!(
        ax3,
        xz,
        M.e2n;
        color=(@. 1 - η55 ./ η55_0),
        colormap=ColorSchemes.vik,
        shading=false,
    )

    # m3.colorrange = extrema(log10.(ηiso.val))

    # Plot -> η isotropy
    ax4 = fig[2, 3] = Axis(fig)
    m4 = mesh!(
        ax4, xz, M.e2n; color=log10.(ηiso.val), colormap=ColorSchemes.vik, shading=false
    )
    m4.colorrange = extrema(log10.(ηiso.val))

    linkaxes!(ax1, ax2)
    linkaxes!(ax3, ax4)

    xlims!(ax2, (0, 2.12))
    ylims!(ax2, (0, 2.12))
    xlims!(ax3, (0, 2.12))
    ylims!(ax3, (-2.12, 0))
    xlims!(ax5, (-2.12, 0))
    ylims!(ax5, (0, 2.12))
    xlims!(ax6, (-2.12, 0))
    ylims!(ax6, (-2.12, 0))

    hidedecorations!(ax1; ticklabels=false)
    hidedecorations!(ax2; ticklabels=false)
    hidedecorations!(ax3; ticklabels=false)
    hidedecorations!(ax4; ticklabels=false)
    hidedecorations!(ax5; ticklabels=false)
    hidedecorations!(ax6; ticklabels=false)

    hideydecorations!(ax1)
    hideydecorations!(ax2)
    hideydecorations!(ax3)
    hideydecorations!(ax4)
    hidexdecorations!(ax1)
    hidexdecorations!(ax2)
    hidexdecorations!(ax5)

    ax1.aspect = DataAspect()
    ax2.aspect = DataAspect()
    ax3.aspect = DataAspect()
    ax4.aspect = DataAspect()
    ax5.aspect = DataAspect()
    ax6.aspect = DataAspect()

    cbarT = Colorbar(fig[1, 4], m1; label="T", width=25, fontsize=6)
    cbarUθ = Colorbar(fig[2, 4], m3; label="Anisotropy", width=25, fontsize=6)
    m3.colorrange = (0.1, 1.0)

    cbarRadial = Colorbar(fig[1, 5], m5; label="Radial anisotropy", width=25, fontsize=6)
    # m5.colorrange = (1.0, 2.5)

    cbar_n55 = Colorbar(fig[2, 5], m6; label="Azimuthal anisotropy", width=25)
    # m6.colorrange = (0, 0.015)

    trim!(fig.layout)
    return fig
end

FSE = rebuild_FSE(vx1, vx2, vy1, vy2, a1, a2)

eigFSE(F::FiniteStrainEllipsoid{T}) where {T} = ([F.x1 F.x2; F.y1 F.y2], [F.a1, F.a1])

function plot_ip(field, area_el)
    fname = "/home/albert/Documents/JuM2TRI/JuM2TRI/JuliaFEM/output/IsoviscousIsotropic_1e4_full_tensor_2/file_1.h5"
    # fname = "/home/albert/Documents/JuM2TRI/JuM2TRI/JuliaFEM/output/IsoviscousIsotropic_1e4_full_tensor/file_50.h5"

    M, V, U, C, Fxx, Fzz, Fxz, Fzx, a1, a2, vx1, vx2, vy1, vy2 = loader(fname)

    fname = "/home/albert/Documents/JuM2TRI/JuM2TRI/JuliaFEM/output/IsoviscousAnisotropic_1e4_benchmark_3_plumes_nullspace2/file_20.h5"
    # U, T = loader_lite(fname)

    M, V, U, C, Fxx, Fzz, Fxz, Fzx, a1, a2, vx1, vx2, vy1, vy2 = loader(fname, false)

    ## Plot
    xz = [M.x M.z]

    a = @. atand(vy1 / vx1)
    # a = @. atand(vy1./vx1)
    # idx = a.<0
    # a[idx].+=360
    Vn = ip2node(M.e2n', area_el, C.η35)
    # Vn = ip2node(M.e2n', area_el, a)

    fig = Figure(; resolution=(1700, 900), backgroundcolor=RGBf0(0.98, 0.98, 0.98))
    ax = fig[1, 1] = Axis(fig)
    m = mesh!(ax, xz[1:64000, :], M.e2n; color=Vn, colormap=:Spectral, shading=false)
    ax.aspect = DataAspect()
    cb = Colorbar(fig[1, 2], m; width=25)
    trim!(fig.layout)
    return fig
end

# i1 =  findall((ipx .>=0.72) .& (ipx.<0.78) .& (ipz .>=1.56) .& (ipz.<1.58))

function plot_node(Vn)
    fname = "/home/albert/Documents/JuM2TRI/JuM2TRI/JuliaFEM/output/IsoviscousIsotropic_1e4_full_tensor_2/file_2.h5"
    # fname = "/home/albert/Documents/JuM2TRI/JuM2TRI/JuliaFEM/output/IsoviscousAnisotropic_1e6/file_210.h5"
    M, V, U, 𝓒, Fxx, Fzz, Fxz, Fzx, a1, a2, vx1, vx2, vy1, vy2 = loader(fname)

    fname = "/home/albert/Documents/JuM2TRI/JuM2TRI/JuliaFEM/output/IsoviscousAnisotropic_1e4_benchmark_3_plumes_nullspace/file_75.h5"
    fname = "/home/albert/Documents/JuM2TRI/JuM2TRI/JuliaFEM/output/IsoviscousAnisotropic_1e4_benchmark_4_plumes_nullspace2/file_50.h5"

    path = "/home/albert/Documents/JuM2TRI/NoAvx/output"
    fname = string(
        pwd(),
        "/output/IsoviscousAnisotropic_1e4_benchmark_4_plumes_GC_longrun_rsquared/",
        "file_42.h5",
    )

    # fname = string(pwd(),
    #                "/output/",
    #                "file_99.h5")

    # fname = "/home/albert/Desktop/file_73.h5"
    M, V, U, 𝓒, Fxx, Fzz, Fxz, Fzx, a1, a2, vx1, vx2, vy1, vy2 = loader(
        fname; isotropic=true
    )
    nT = M.e2n[end, 6]

    v1 = @. sqrt(U.x[1:nT]^2 + U.z[1:nT]^2)
    v2 = @. sqrt(U.θ[1:nT]^2 + U.r[1:nT]^2)

    # fname = "/home/albert/Desktop/TaylorHood/output/IsoviscousIsotropic_1e4_TaylorHood/file_15.h5"
    # U,T = loader_lite(fname)

    ## Plot
    xz = [M.x M.z]

    fig = Figure(; resolution=(1700, 900), backgroundcolor=RGBf0(0.98, 0.98, 0.98))

    # Plot -> Radial anisotropy
    ax = fig[1, 1] = Axis(fig)
    # m = mesh!(ax, xz[1:64000,:], M.e2n, color = Vn, colormap = :Spectral, shading = false) 
    # m = mesh!(ax, xz[1:nT,:], M.e2n, color = U.r[1:nT], colormap = :Spectral, shading = false) 
    m = mesh!(ax, xz[1:nT, :], M.e2n; color=U.θ[1:nT], colormap=:Spectral, shading=false)
    # m = mesh!(ax, xz[1:64000,:], M.e2n, color = Ur[1:64000], colormap = :Spectral, shading = false) 
    # m = mesh!(ax, xz[1:64000,:], M.e2n, color = U.x[1:64000], colormap = :Spectral, shading = false) 
    m = mesh!(ax, xz[1:nT, :], M.e2n; color=V.T[1:nT], colormap=:Spectral, shading=false)
    # m = mesh!(ax, xz[1:nT,:], M.e2n, color = T[1:nT], colormap = :Spectral, shading = false)
    # scaling = 1e-3
    # arrows!(xz[:,1], xz[:,2], U.x[1:64000].*scaling, U.z[1:64000].*scaling, arrowsize=1e-3, linecolor=:blue)

    ax.aspect = DataAspect()

    cb = Colorbar(fig[1, 2], m; width=25)

    trim!(fig.layout)
    return fig
end

fname = "/home/albert/Documents/JuM2TRI/JuM2TRI/JuliaFEM/DEM/DEM_1e-3_vol20.h5"
D = getDEM(fname)

idx = ipx .> 2π
ipx[idx] .-= 2π

str = "/home/albert/Desktop/Persephone2Cluster/ClusterOutput/"
fname = joinpath(str, "Anisotropic/Isoviscous/LinearT/1e6/file_500.h5")

fname1 = "/home/albert/Desktop/OUTPUT/Persephone/Ani1e6_Perturbed/file_17.h5"
fname2 = "/home/albert/Documents/JuM2TRI/JuM2TRI/JuliaFEM/output/IsoviscousIsotropic_1e4/file_26.h5"

# fname1 = "/home/albert/Documents/JuM2TRI/JuM2TRI/JuliaFEM/output/Anisotropic1e6_Tdep/file_120.h5"
# fname2 = "/home/albert/Documents/JuM2TRI/JuM2TRI/JuliaFEM/output/Isotropic1e6_Tdep/file_120.h5"

fname1 = "/home/albert/Desktop/OUTPUT/Persephone/Anisotropic_Isoviscous1e5/file_237.h5"
fname2 = "/home/albert/Desktop/OUTPUT/Persephone/Isotropic_Isoviscous1e5/file_139.h5"

make_plot(fname1, fname2, ipx, D)

# xy = [Point2D{Cartesian}(ipx[i], ipz[i]) for i in eachindex(ipx)]
# Uxg = togrid(xg, yg, xy, vx1)

# X = xg'.*ones(length(yg))
# Y = yg.*ones(1,length(xg))

# Uxg, Uzg, xg, zg = structured_velocity(M.θ, M.r, Uiso)

# strf = streamline(Uxg', Uzg', xg, zg )
# xc, zc = polar2cartesian(xg,zg)
# contour(xg[1,:],zg[:,1],strf,levels=20)

scatter(M.x[M.e2n[1, :]], M.z[M.e2n[1, :]])

scatter!([M.x[M.e2n[1, 1]]], [M.z[M.e2n[1, 1]]]; color=:red)
scatter!([M.x[M.e2n[1, 2]]], [M.z[M.e2n[1, 2]]]; color=:green)
scatter!([M.x[M.e2n[1, 3]]], [M.z[M.e2n[1, 3]]]; color=:blue)
scatter!([M.x[M.e2n[1, 4]]], [M.z[M.e2n[1, 4]]]; color=:black)
scatter!([M.x[M.e2n[1, 5]]], [M.z[M.e2n[1, 5]]]; color=:orange)
scatter!([M.x[M.e2n[1, 6]]], [M.z[M.e2n[1, 6]]]; color=:yellow)
