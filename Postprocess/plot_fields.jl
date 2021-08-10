include(joinpath(pwd(), "Plotting/misc.jl"))
using DelimitedFiles

function plot_stats(fstats)
    A = readdlm(fstats)
    t = A[:,1]
    isort = sortperm(t)
    sort!(t)
    rms = A[isort,2]
    Tav = A[isort,3]
    Nut = A[isort,5]
    Nub = A[isort,4]
    
    fig = Figure(resolution = (1600, 900), backgroundcolor = RGBf0(1, 1, 1))

    # Urms
    ax1 = fig[1, 1] = Axis(fig)
    lines!(ax1, t, rms)
    ax1.ylabel = "Urms"

    # ⟨T⟩
    ax2 = fig[2, 1] = Axis(fig)
    lines!(ax2, t, Tav)
    ax2.ylabel = "⟨T⟩"

    # Urms
    ax3 = fig[3, 1] = Axis(fig)
    lines!(ax3, t, Nut, color=:blue, label = "top")
    lines!(ax3, t, Nub, color=:red, label = "top")
    ax3.ylabel = "Nusselt"
    ax3.xlabel = "t"

    lims = extrema(t)
    xlims!(ax1, lims)
    xlims!(ax2, lims)
    xlims!(ax3, lims)

    hidexdecorations!(ax1)
    hidexdecorations!(ax2)

    fig
end

   
function plot_anisotropy(fname, gr; isave = "no")

    M, V, U, 𝓒, Fxx, Fzz, Fxz, Fzx, a1, a2, vx1, vx2, vy1, vy2 = loader(fname, isotropic=false)
    Ael = element_area(gr)

    FSE = [FiniteStrainEllipsoid{Float64}(1.0, 0.0, 0.0, 1.0, a1[i], a2[i]) 
                    for i in CartesianIndices(vx1)]

    fdem = joinpath("/home/albert/Desktop/CylParticles/DEM", 
                    "DEM_1e-3_vol20_new3.h5")
    D = getDEM(fdem)

    # Isotropic η55 
    η55_0 = maximum(D.𝓒[:,5])

    𝓒 = anisotropic_tensor_postprocess(FSE, D);

    η11 = [𝓒[i][1,1] for i in CartesianIndices(𝓒)]
    η22 = [𝓒[i][2,2] for i in CartesianIndices(𝓒)]
    η12 = [𝓒[i][1,2] for i in CartesianIndices(𝓒)]
    η45 = [𝓒[i][4,5] for i in CartesianIndices(𝓒)]
    η44 = [𝓒[i][4,4] for i in CartesianIndices(𝓒)]
    η55 = [𝓒[i][5,5] for i in CartesianIndices(𝓒)]
    η66 = [𝓒[i][6,6] for i in CartesianIndices(𝓒)]
    anisotropy = @. 1-η55/η55_0

    # normalise anisotropy
    ω = anisotropy./maximum(anisotropy)
    anisotropy_normalised = ip2node(gr.e2n, Ael, ω) 
    applybounds!(anisotropy_normalised, 1.0, 0.0)

    # Radial anisotropy 
    N = @. 0.125*(η11 + η22) - 0.25*η12 + 0.5*η66
    L = @. 0.5*(η44 + η55)
    χ = N./L
    χn = ip2node(gr.e2n, Ael, χ)

    # Azimuthal anisotropy
    Gc = @. 0.5*(η55 - η44)
    Gs = η45
    G = @. 0.5*(Gc^2 + Gs^2)
    Gn = ip2node(gr.e2n, Ael, G)
    Gn = max.(Gn, 0.0)

    xz = hcat(gr.x, gr.z)

    fig = Figure(resolution = (1225, 900), backgroundcolor = RGBf0(1, 1, 1))

    # T
    ax = fig[1, 2] = Axis(fig)
    m = mesh!(ax, xz, gr.e2n_p1', color = V.T, colormap = Reverse(:Spectral), shading = false)
    
    ax.aspect = DataAspect()
    hidedecorations!(ax)
    hidespines!(ax)
    xlims!(ax, -2.22, 0)
    ylims!(ax, 0, 2.22)
    
    Colorbar(fig[1, 1], m, width=25, label = "T", height = Relative(3/4), flipaxis=false)

    # Radial anisotropy
    ax1 = fig[1, 3] = Axis(fig)
    m1 = mesh!(ax1, xz, gr.e2n_p1', color = χn, colormap = Reverse(:watermelon), shading = false)

    ax1.aspect = DataAspect()
    hidedecorations!(ax1)
    hidespines!(ax1)
    xlims!(ax1, 0, 2.22)
    ylims!(ax1, 0, 2.22)

    Colorbar(fig[1, 4], m1, width=25, label = "Radial anisotropy", height = Relative(3/4))

    # Azimuthal anisotropy
    ax2 = fig[2, 3] = Axis(fig)
    m2 = mesh!(ax2, xz, gr.e2n_p1', color = Gn, colormap = Reverse(:Spectral), shading = false)

    ax2.aspect = DataAspect()
    hidedecorations!(ax2)
    hidespines!(ax2)
    xlims!(ax2, 0, 2.22)
    ylims!(ax2, -2.22, 0)

    Colorbar(fig[2, 4], m2, width=25, label = "Azimuthal anisotropy", height = Relative(3/4))

    # Anisotropy
    ax3 = fig[2, 2] = Axis(fig)
    m3 = mesh!(ax3, xz, gr.e2n_p1', color = anisotropy_normalised, colormap = Reverse(:Spectral), shading = false)
    
    ax3.aspect = DataAspect()
    hidedecorations!(ax3)
    hidespines!(ax3)
    xlims!(ax3, -2.22, 0)
    ylims!(ax3, -2.22, 0)
    
    Colorbar(fig[2, 1], m3, width=25, label = "1-η55/η", height = Relative(3/4), flipaxis=false)

    trim!(fig.layout)

    if isave != "no"
        save(isave, fig)
    end

    fig
end

function plot_node(Vn, M)

    xz = [M.x M.z]
    nT = M.e2n[end,6]

    fig = Figure(resolution = (1100, 900), backgroundcolor = RGBf0(1, 1, 1))

    # Plot field
    ax = fig[1, 1] = Axis(fig)
    m = mesh!(ax, xz[1:nT,:], M.e2n, color = Vn[1:nT], colormap = Reverse(:Spectral), shading = false)
    
    ax.aspect = DataAspect()
    hideydecorations!(ax)
    hidespines!(ax)
    xlims!(ax, -2.22, 2.22)
    ylims!(ax, -2.22, 2.22)
    
    Colorbar(fig[1, 2], m, width=25, label = "T")
    
    trim!(fig.layout)
    fig

end

function plot_node(Vn, gr::Grid)
    xz = hcat(gr.x, gr.z)
    fig = Figure(resolution = (1100, 900), backgroundcolor = :white)

    clim = maximum(abs.(extrema(Vn)))

    # Plot field
    ax = fig[1, 1] = Axis(fig, backgroundcolor = :white)
    m = mesh!(ax, xz, gr.e2n_p1', 
            #   colorrange= (0,0.01),
              color = Vn, 
              colormap = Reverse(:Spectral), 
              shading = false,
              backgroundcolor = RGBf0(0,0,0))
                
    ax.aspect = DataAspect()
    
    xlims!(ax, -2.22, 2.22)
    ylims!(ax, -2.22, 2.22)

    hidedecorations!(ax)
    hidespines!(ax)
    
    Colorbar(fig[1, 2], m, width=25, label = "T")
    
    trim!(fig.layout)
    fig

end

function plot_node_black(Vn, gr::Grid)
    xz = hcat(gr.x, gr.z)
    
    clim = maximum(abs.(extrema(Vn)))
    
    # Plot field
    fig = Figure(resolution = (1100, 900), backgroundcolor = :black)
    ax = fig[1, 1] = Axis(fig, backgroundcolor = :black)
    m = mesh!(ax, xz, gr.e2n_p1', 
            #   colorrange= (0,0.01),
              color = Vn, 
              colormap = Reverse(:Spectral), 
              shading = false)
                
    ax.aspect = DataAspect()
    xlims!(ax, -2.22, 2.22)
    ylims!(ax, -2.22, 2.22)

    hidedecorations!(ax)
    hidespines!(ax)
    
    Colorbar(fig[1, 2], m, width=25, label = "T", labelcolor = :white, tickcolor = :white, ticklabelcolor = :white)
    
    trim!(fig.layout)
    fig

end

function sortfiles(files::Vector{String})
    idx = occursin.(".h5", files)
    nf = sum(idx)
    ii = findall(idx)
    num = Vector{Int32}(undef, nf)
    for i in ii
        @inbounds ifirst = findall("file_", files[i])[1][end]+1
        @inbounds num[i] = parse(Int32, files[i][ifirst:end-3])
    end
    files[sortperm(num)]
end

function animation_node_black(gr::Grid, fname::String)
    # get files
    files = readdir(fname, join=true)
    files = sortfiles(files)
    # load 1st file
    M, V, U, 𝓒, Fxx, Fzz, Fxz, Fzx, a1, a2, vx1, vx2, vy1, vy2 = loader(files[1], isotropic=false);

    Vn = V.T

    xz = hcat(gr.x, gr.z)
    
    clim = maximum(abs.(extrema(Vn)))
    
    # Plot field
    fig = Figure(resolution = (1100, 900), backgroundcolor = :black)
    ax = fig[1, 1] = Axis(fig, backgroundcolor = :black)
    m = mesh!(ax, xz, gr.e2n_p1', 
            #   colorrange= (0,0.01),
              color = Vn, 
              colormap = Reverse(:Spectral), 
              shading = false)
                
    ax.aspect = DataAspect()
    xlims!(ax, -2.22, 2.22)
    ylims!(ax, -2.22, 2.22)

    hidedecorations!(ax)
    hidespines!(ax)
    
    Colorbar(fig[1, 2], m, width=25, label = "T", labelcolor = :white, tickcolor = :white, ticklabelcolor = :white)
    
    trim!(fig.layout)
    fig

    record(fig, "Temperature.mp4", files; framerate = 30) do fl
        M, V, U, 𝓒, Fxx, Fzz, Fxz, Fzx, a1, a2, vx1, vx2, vy1, vy2 = loader(fl, isotropic=false);
        m.color = V.T
    end

end


function resample_fields(gr, xg, yg, ipx, ipz, vx1, vy1, U)
    # Resample eigenvectors and velocity onto structured grid
    vx1g = togrid(xg,yg,vec(ipx),vec(ipz),vec(vx1))
    vz1g = togrid(xg,yg,vec(ipx),vec(ipz),vec(vy1))

    Ux = togrid(xg, yg, gr.θ, gr.r, U.x)
    Uz = togrid(xg, yg, gr.θ, gr.r, U.z)

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
    vx1g[:,1] = vx1g[:,end] = mean([vx1g[:,1], vx1g[:,end]])
    vz1g[:,1] = vz1g[:,end] = mean([vx1g[:,1], vx1g[:,end]])
    Ux[:,1] = Ux[:,end] = mean([vx1g[:,1], vx1g[:,end]])
    Uz[:,1] = Uz[:,end] = mean([vx1g[:,1], vx1g[:,end]])

    return vx1g, vz1g, Ux, Uz
end


function plot_4velocities(gr, fname::Vector{String})

    _, _, U1, = loader(fname[1], isotropic=false);
    _, _, U2, = loader(fname[2], isotropic=false);
    _, _, U3, = loader(fname[3], isotropic=false);
    _, _, U4, = loader(fname[4], isotropic=false);
    scaling = 1e3

    fig = Figure(resolution = (1100, 1100), backgroundcolor = :black)

    # TOP LEFT CORNER
    ax1 = fig[1, 1] = Axis(fig, backgroundcolor = :black)
    ar1 = arrows!(
        ax1,
        gr.x, gr.z,
        U1.x/scaling, U1.z/scaling,
        arrowsize = 2.5e-2, 
        arrowcolor = :white, 
        linecolor = :white, 
        linewidth =2
    )

    ax1.aspect = DataAspect()
    xlims!(ax1, -2.22, 0)
    ylims!(ax1, 0, 2.22)

    hidedecorations!(ax1)
    hidespines!(ax1)

    # TOP RIGHT CORNER
    ax2 = fig[1, 2] = Axis(fig, backgroundcolor = :black)
    ar2 = arrows!(
        ax2,
        gr.x, gr.z,
        U2.x/scaling, U2.z/scaling,
        arrowsize = 2.5e-2, 
        arrowcolor = :white, 
        linecolor = :white,
        linewidth =2
    )

    ax2.aspect = DataAspect()
    xlims!(ax2, 0, 2.22)
    ylims!(ax2, 0, 2.22)

    hidedecorations!(ax2)
    hidespines!(ax2)


    # BOT LEFT CORNER
    ax3 = fig[2, 1] = Axis(fig, backgroundcolor = :black)
    ar3 = arrows!(
        ax3,
        gr.x, gr.z,
        U3.x/scaling, U3.z/scaling,
        arrowsize = 2.5e-2, 
        arrowcolor = :white, 
        linecolor = :white, 
        linewidth =2
    )

    ax3.aspect = DataAspect()
    xlims!(ax3, -2.22, 0)
    ylims!(ax3, -2.22, 0)

    hidedecorations!(ax3)
    hidespines!(ax3)

    # BOT RIGHT CORNER
    ax4 = fig[2, 2] = Axis(fig, backgroundcolor = :black)
    ar4 = arrows!(
        ax4,
        gr.x, gr.z,
        U4.x/scaling, U4.z/scaling,
        arrowsize = 2.5e-2, 
        arrowcolor = :white, 
        linecolor = :white,
        linewidth =2
    )

    ax4.aspect = DataAspect()
    xlims!(ax4, 0, 2.22)
    ylims!(ax4, -2.22, 0)

    hidedecorations!(ax4)
    hidespines!(ax4)

    fig

end