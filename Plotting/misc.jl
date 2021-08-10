using GLMakie
using ColorSchemes
import Statistics: mean

struct Mesh{T,M}
    x::Vector{T}
    z::Vector{T}
    θ::Vector{T}
    r::Vector{T}
    e2n::Array{M,2}
end

struct Var{T}
    T::Vector{T}
    ρ::Vector{T}
    η::Vector{T}
end

struct Vel{T}
    θ::Vector{}
    r::Vector{T}
    x::Vector{T}
    z::Vector{T}
end

function cartesian2polar(x, z)
    return atan(x, z), sqrt(x^2 + z^2)
end

function polar2cartesian(x::Array, z::Array)
    a = @. z * sin(x)
    b = @. z * cos(x)
    return a, b
end

function elcenter(x, z, e2n)
    n = size(e2n, 1)
    xc = Vector{Float64}(undef, n)
    zc = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        el = e2n[i, :]
        xc[i] = mean(view(x, el))
        zc[i] = mean(view(z, el))
    end
    return xc, zc
end

function quadrants(θ, r, e2n)
    θc, rc = elcenter(θ, r, e2n)
    q1 = findall(x -> 0 <= x < π / 2, θc)
    q2 = findall(x -> π / 2 < x < π, θc)
    q3 = findall(x -> π < x < 3π / 2, θc)
    q4 = findall(x -> 3π / 2 < x < 2π, θc)
    Q1 = e2n[q1, :]
    Q2 = e2n[q2, :]
    Q3 = e2n[q3, :]
    Q4 = e2n[q4, :]
    return Q1, Q2, Q3, Q4
end

## LOAD Variables 
function loadMesh(path_mesh, file="MESH.mat")
    str = joinpath(path_mesh, file)
    file = matopen(str)
    MESH = read(file, "MESH")
    xy = get(MESH, "GCOORD", 1)
    EL2NOD = 1 * get(MESH, "EL2NOD", 1)
    EL2NOD_P1 = 1 * get(MESH, "EL2NOD_P1", 1)
    thr = get(MESH, "GCOORD_POL", 1)
    θ = thr[1, :]
    r = thr[2, :]
    x = xy[1, :]
    z = xy[2, :]
    return Mesh(x, z, θ, r, EL2NOD)
end

function loader_lite(fname)
    fid = h5open(fname, "r")

    # -- Load fields
    V = fid["VAR"]
    Ux = read(V, "Ux")
    Uz = read(V, "Uz")
    Uθ = read(V, "Utheta")
    Ur = read(V, "Ur")
    U = Vel(Uθ, Ur, Ux, Uz)
    T = read(V, "T")
    return U, T
end

function loader(fname; isotropic=true, getTime=true)
    fid = h5open(fname, "r")

    # -- Load time
    if getTime == true
        Time = fid["Time"]
        t = read(Time, "t")
    else
        t = nothing
    end

    # -- Load mesh
    M = fid["MESH"]
    θ = read(M, "x")
    r = read(M, "z")
    e2n = read(M, "EL2NOD")
    x, z = polar2cartesian(θ, r)
    M = Mesh(x, z, θ, r, Array(e2n'))

    # -- Load fields
    V = fid["VAR"]
    Ux = read(V, "Ux")
    Uz = read(V, "Uz")
    Uθ = read(V, "Utheta")
    Ur = read(V, "Ur")
    ρ = read(V, "density")
    η = read(V, "nu")
    # T = dropdims(read(V,"T"),dims=2)
    T = read(V, "T")
    if (η isa Vector) == false
        η = fill(η, length(T))
    end

    Fxx = read(V["Fxx"])
    Fzz = read(V["Fzz"])
    Fxz = read(V["Fxz"])
    Fzx = read(V["Fzx"])

    a1 = read(V["a1"])
    a2 = read(V["a2"])
    vx1 = read(V["x1"])
    vx2 = read(V["x2"])
    vy1 = read(V["y1"])
    vy2 = read(V["y2"])

    if isotropic == false
        η11 = read(V, "nu11")
        η33 = read(V, "nu33")
        η55 = read(V, "nu55")
        η13 = read(V, "nu13")
        η15 = read(V, "nu15")
        η35 = read(V, "nu35")
        𝓒 = StiffnessTensor(η11, η33, η55, η13, η15, η35)
    else
        𝓒 = nothing
    end

    V = Var(T, ρ, η)
    U = Vel(Uθ, Ur, Ux, Uz)

    return M, V, U, 𝓒, Fxx, Fzz, Fxz, Fzx, a1, a2, vx1, vx2, vy1, vy2, t
end

## Grid Interpolants
function togrid(xg, yg, xp, yp, f)
    nodes_i = Vector{Int64}(undef, 2)
    nodes_j = Vector{Int64}(undef, 2)
    ω = Array{Float64,2}(undef, 2, 2)

    xnum, ynum = length(xg), length(yg)
    x0, y0 = xg[1], yg[1]
    Δx, Δy = xg[2] - xg[1], yg[2] - yg[1]
    X = xg' .* ones(length(yg))
    Y = yg .* ones(1, length(xg))
    gridfield_up = fill(0.0, ynum, xnum)
    gridfield_down = fill(0.0, ynum, xnum)

    for i in axes(xp, 1)
        xpᵢ = xp[i] - x0
        ypᵢ = min(yp[i] - y0, 1.0)

        xcell = Int(div(xpᵢ, Δx)) + 1 # Indexing could be more
        ycell = Int(div(ypᵢ, Δy)) + 1 # efficient to avoid 
        # allocation of X and Y
        # if xcell == 0
        #     xcell = 1
        # end
        # if ycell == 0
        #     ycell = 1
        # end

        # println(xcell)
        # println(ycell)
        # println(xpᵢ)
        # println(ypᵢ)

        nodes_j[1] = xcell
        nodes_j[2] = xcell + 1 > length(xg) ? 1 : xcell + 1 # array circularity
        nodes_i[1] = ycell
        nodes_i[2] = ycell + 1 > length(yg) ? ycell : ycell + 1

        nodes_x = view(X, nodes_i, nodes_j)
        nodes_y = view(Y, nodes_i, nodes_j)

        ω .= 1.0 ./ (@. sqrt((xpᵢ - nodes_x)^2 + (ypᵢ - nodes_y)^2))

        gridfield_up[nodes_i, nodes_j] .+= ω * f[i]
        gridfield_down[nodes_i, nodes_j] .+= ω
    end

    return gridfield_up ./ gridfield_down
end

function togrid(xg, yg, P::Array{Point2D{Polar},1}, f)
    nodes_i = Vector{Int64}(undef, 2)
    nodes_j = Vector{Int64}(undef, 2)
    ω = Array{Float64,2}(undef, 2, 2)

    xnum, ynum = length(xg), length(yg)
    x0, y0 = xg[1], yg[1]
    Δx, Δy = xg[2] - xg[1], yg[2] - yg[1]
    X = xg' .* ones(length(yg))
    Y = yg .* ones(1, length(xg))
    gridfield_up = fill(0.0, xnum, ynum)
    gridfield_down = fill(0.0, xnum, ynum)

    for i in axes(P, 1)
        xpᵢ = Point2D{Polar}(P[i].x - x0, P[i].z - y0)

        xcell = Int(div(xpᵢ.x, Δx, RoundUp)) # Indexing could be more
        ycell = Int(div(xpᵢ.z, Δy, RoundUp)) # efficient to avoid 
        # allocation of X and Y

        if xcell == 0
            xcell = 1
        end
        if ycell == 0
            ycell = 1
        end
        nodes_j[1] = xcell
        nodes_j[2] = xcell + 1
        nodes_i[1] = ycell
        nodes_i[2] = ycell + 1

        nodes = Point2D{Polar}.(view(X, nodes_i, nodes_j), view(Y, nodes_i, nodes_j))

        ω .= [
            1.0 ./ (@. sqrt((xpᵢ.x - nodes[u].x)^2 + (xpᵢ.z - nodes[u].z)^2)) for
            u in CartesianIndices(nodes)
        ]

        gridfield_up[nodes_i, nodes_j] .+= ω * f[i]
        gridfield_down[nodes_i, nodes_j] .+= ω
    end

    return gridfield_up ./ gridfield_down
end

function togrid(xg, yg, P::Array{Point2D{Cartesian},1}, f)
    nodes_i = Vector{Int64}(undef, 2)
    nodes_j = Vector{Int64}(undef, 2)
    ω = Array{Float64,2}(undef, 2, 2)

    xnum, ynum = length(xg), length(yg)
    x0, y0 = xg[1], yg[1]
    Δx, Δy = xg[2] - xg[1], yg[2] - yg[1]
    X = xg' .* ones(length(yg))
    Y = yg .* ones(1, length(xg))
    gridfield_up = fill(0.0, xnum, ynum)
    gridfield_down = fill(0.0, xnum, ynum)

    @inbounds for i in axes(P, 1)
        xpᵢ = Point2D{Cartesian}(P[i].x - x0, P[i].z - y0)

        xcell = Int(div(xpᵢ.x, Δx, RoundUp)) # Indexing could be more
        ycell = Int(div(xpᵢ.z, Δy, RoundUp)) # efficient to avoid 
        # allocation of X and Y

        if xcell == 0
            xcell = 1
        end
        if ycell == 0
            ycell = 1
        end

        nodes_j[1] = xcell
        nodes_j[2] = xcell + 1
        nodes_i[1] = ycell
        nodes_i[2] = ycell + 1

        nodes = Point2D{Cartesian}.(view(X, nodes_i, nodes_j), view(Y, nodes_i, nodes_j))

        ω .= [
            1.0 ./ (@. sqrt((xpᵢ.x - nodes[u].x)^2 + (xpᵢ.z - nodes[u].z)^2)) for
            u in CartesianIndices(nodes)
        ]

        gridfield_up[nodes_i, nodes_j] .+= ω * f[i]
        gridfield_down[nodes_i, nodes_j] .+= ω
    end

    return gridfield_up ./ gridfield_down
end

function topoint(xg, yg, P::Array{Point2D{Polar},1}, f)
    nodes_i = Vector{Int64}(undef, 2)
    nodes_j = Vector{Int64}(undef, 2)
    ω = Array{Float64,2}(undef, 2, 2)

    nP = length(P)
    xnum, ynum = length(xg), length(yg)
    x0, y0 = xg[1], yg[1]
    Δx, Δy = xg[2] - xg[1], yg[2] - yg[1]
    X = xg' .* ones(length(yg))
    Y = yg .* ones(1, length(xg))
    pointfield_up = fill(0.0, nP)
    pointfield_down = fill(0.0, nP)

    @inbounds for i in axes(P, 1)
        xpᵢ = Point2D{Polar}(P[i].x - x0, P[i].z - y0)

        xcell = Int(div(xpᵢ.x, Δx, RoundUp)) # Indexing could be more
        ycell = Int(div(xpᵢ.z, Δy, RoundUp)) # efficient to avoid 
        # allocation of X and Y

        if xcell == 0
            xcell = 1
        end
        if ycell == 0
            ycell = 1
        end
        nodes_j[1] = xcell
        nodes_j[2] = xcell + 1
        nodes_i[1] = ycell
        nodes_i[2] = ycell + 1

        nodes = Point2D{Polar}.(view(X, nodes_i, nodes_j), view(Y, nodes_i, nodes_j))

        ω .= 1.0 ./ (@. sqrt((xpᵢ.x - nodes.x)^2 + (xpᵢ.z - nodes.z)^2))

        pointfield_up[i] += sum(ω .* f[nodes_i, nodes_j])
        pointfield_down[i] += sum(ω)
    end

    return pointfield_up ./ pointfield_down
end

function topoint(xg, yg, P::Array{Point2D{Cartesian},1}, f)
    nodes_i = Vector{Int64}(undef, 2)
    nodes_j = Vector{Int64}(undef, 2)
    ω = Array{Float64,2}(undef, 2, 2)

    nP = length(P)
    xnum, ynum = length(xg), length(yg)
    x0, y0 = xg[1], yg[1]
    Δx, Δy = xg[2] - xg[1], yg[2] - yg[1]
    X = xg' .* ones(length(yg))
    Y = yg .* ones(1, length(xg))
    pointfield_up = fill(0.0, nP)
    pointfield_down = fill(0.0, nP)

    @inbounds for i in axes(P, 1)
        xpᵢ = Point2D{Cartesian}(P[i].x - x0, P[i].z - y0)

        xcell = Int(div(xpᵢ.x, Δx, RoundUp)) # Indexing could be more
        ycell = Int(div(xpᵢ.z, Δy, RoundUp)) # efficient to avoid 
        # allocation of X and Y

        if xcell == 0
            xcell = 1
        end
        if ycell == 0
            ycell = 1
        end
        nodes_j[1] = xcell
        nodes_j[2] = xcell + 1
        nodes_i[1] = ycell
        nodes_i[2] = ycell + 1

        nodes = Point2D{Cartesian}.(view(X, nodes_i, nodes_j), view(Y, nodes_i, nodes_j))

        ω .= 1.0 ./ (@. sqrt((xpᵢ.x - nodes.x)^2 + (xpᵢ.z - nodes.z)^2))

        pointfield_up[i] += sum(ω .* f[nodes_i, nodes_j])
        pointfield_down[i] += sum(ω)
    end

    return pointfield_up ./ pointfield_down
end

function streamlines(x, y, U, P, nIts=10, coordinates="Polar")
    xg = LinRange(minimum(x), maximum(x), 100)
    yg = LinRange(minimum(y), maximum(y), 100)

    if coordinates == "Cartesian"
        xy = [Point2D{Cartesian}(x[i], y[i]) for i in axes(x, 1)]
    elseif coordinates == "Polar"
        xy = [Point2D{Polar}(x[i], y[i]) for i in axes(x, 1)]
    end

    Uxg = togrid(xg, yg, xy, U.x)
    Uzg = togrid(xg, yg, xy, U.z)

    return streamliner(P, Uxg, Uzg, nIts)
end

function structured_velocity(x, y, U, coordinates="Polar")
    xg = LinRange(minimum(x), maximum(x), 150)
    yg = LinRange(minimum(y), maximum(y), 100)

    if coordinates == "Cartesian"
        xy = [Point2D{Cartesian}(x[i], y[i]) for i in axes(x, 1)]
        Uxg = togrid(xg, yg, xy, U.x)
        Uzg = togrid(xg, yg, xy, U.z)

    elseif coordinates == "Polar"
        xy = [Point2D{Polar}(x[i], y[i]) for i in axes(x, 1)]
        Uxg = togrid(xg, yg, xy, U.θ)
        Uzg = togrid(xg, yg, xy, U.r)
    end

    X = xg' .* ones(length(yg))
    Y = yg .* ones(1, length(xg))

    return Uxg, Uzg, X, Y
end

function streamliner(P, Uxg, Uzg, nIts)
    Δx, Δy = xg[2] - xg[1], yg[2] - yg[1]
    x, z = [Float64[] for i in axes(P, 1)], [Float64[] for i in axes(P, 1)]

    @inbounds for _ in 1:Int(nIts)
        # Interpolate velocity to points
        Uxp = topoint(xg, yg, P, Uxg)
        Uzp = topoint(xg, yg, P, Uzg)
        Velp = Point2D{Polar}.(Uxp, Uzp)

        # Compute Δt
        Umax = maximum(max(Uxp, Uzp))
        Δt = min(Δx, Δy) * 0.5 / Umax

        # Advect points
        P = P + Velp * Δt

        # Save lines
        x = [push!(x[i], P[i].x) for i in axes(P, 1)]
        z = [push!(z[i], P[i].z) for i in axes(P, 1)]
    end
    return x, z
end

# function streamline(x,z,ux,uz)
#     streamf = fill(0.0,size(x,1),size(x,1))
#     #  Integrate VZ along X
#     for k in axes(x,1), j in axes(x,2)
#             streamf[k,j] = if j==1
#                 uz[k,j]*(x[j+1]-x[j]);
#             else
#                 streamf[k,j-1]+uz[k,j-1]*(x[j]-x[j-1]);
#             end

#     end
#     # Add integration of Vx along Z
#     for k in axes(x,1), j in axes(x,2)
#         streamf[k,j] = if k == 1
#             streamf[k,j] - ux[k,j]*(z[k+1]-z[k]);
#         else
#             streamf[k-1,j] - ux[k-1,j]*(z[k]-z[k-1])
#         end
#     end
#     streamf
# end

function streamline(x, z, ux, uz)
    streamf = fill(0.0, size(x, 1), size(x, 1))
    #  Integrate VZ along X
    for k in axes(x, 1), j in axes(x, 2)
        streamf[k, j] = if j == 1
            uz[k, j] * (x[j + 1] - x[j]) / z[k, j]
        else
            (streamf[k, j - 1] + uz[k, j - 1] * (x[j] - x[j - 1])) / z[k, j]
        end
    end
    # Add integration of Vx along Z
    for k in axes(x, 1), j in axes(x, 2)
        streamf[k, j] = if k == 1
            streamf[k, j] - ux[k, j] * (z[k + 1] - z[k])
        else
            streamf[k - 1, j] - ux[k - 1, j] * (z[k] - z[k - 1])
        end
    end
    return streamf
end

# distance(x,y,X,Y) = sqrt((x-X)^2+(y-Y)^2)
speed(x, y) = sqrt(x^2 + y^2)

function anisotropic_tensor_postprocess(
    FSE::Array{FiniteStrainEllipsoid{Float64},2}, D::DEM
)
    # -- Allocate arrays
    C = [fill(0.0, 6, 6) for i in CartesianIndices(FSE)]
    v₁ = [similar(D.a1a2_blk) for i in 1:Threads.nthreads()]
    v₂ = [similar(D.a2a3_blk) for i in 1:Threads.nthreads()]
    max_a1a2, max_a2a3 = maximum(D.a1a2_blk), maximum(D.a2a3_blk)
    # -- Fitting coefficients of the axes parameterisation
    R1, R2 = fittingcoefficients()
    # -- Get η from data base and rotate it
    Threads.@threads for i in eachindex(FSE)
        get_full_tensor!(C, FSE[i], R1, R2, D, i, v₁, v₂, max_a1a2, max_a2a3)
    end

    return C
end ### END rotate_tensor FUNCTION #############################################

function get_full_tensor!(C, FSEᵢ, R1, R2, D, i, v₁, v₂, max_a1a2, max_a2a3)
    ## GET VISCOUS TENSOR ========================================================
    # Average fabric -> r₁ = log10(a1/a2) and r₂ = log10(a2/a3)
    r₁, r₂ = fabric_parametrisation(FSEᵢ.a1, 1.0, 1 / FSEᵢ.a2, R1, R2)
    nt = Threads.threadid()

    if r₁ > max_a1a2
        r₁_imin = 49
    else
        @inbounds for j in eachindex(D.a1a2_blk)
            v₁[nt][j] = abs(r₁ - D.a1a2_blk[j])
        end
        r₁_imin = argminsorted(v₁[nt])
    end

    if r₂ > max_a2a3
        r₂_imin = 1
    else
        @inbounds for j in eachindex(D.a2a3_blk)
            v₂[nt][j] = abs(r₂ - D.a2a3_blk[j])
        end
        r₂_imin = D.permutation_blk[argminsorted(v₂[nt])]
    end

    im = D.sblk * (r₁_imin - 1) + r₂_imin
    # Allocate stiffness tensor
    C[i][1, 1] = D.𝓒[im, 1]
    C[i][2, 2] = D.𝓒[im, 2]
    C[i][3, 3] = D.𝓒[im, 3]
    C[i][4, 4] = D.𝓒[im, 4]
    C[i][5, 5] = D.𝓒[im, 5]
    C[i][6, 6] = D.𝓒[im, 6]
    C[i][1, 2] = C[i][2, 1] = D.𝓒[im, 9]
    C[i][1, 3] = C[i][3, 1] = D.𝓒[im, 8]
    return C[i][2, 3] = C[i][3, 2] = D.𝓒[im, 7]

    # C[i] .=  [D.𝓒[im, 1]  D𝓒[im, 9]  D.𝓒[im, 8]  0          0           0
    #          D.𝓒[im, 9]  D.𝓒[im, 2]  D.𝓒[im, 7]  0          0           0
    #          D.𝓒[im, 8]  D.𝓒[im, 7]  D.𝓒[im, 3]  0          0           0
    #          0           0           0           D.𝓒[im, 4] 0           0 
    #          0           0           0           0          D.𝓒[im, 5]  0
    #          0           0           0           0          0           D.𝓒[im, 6]]
    ## ============================================================================

end

function cartesian_tensor(C, vx1, vx2, vy1, vy2, ipx)
    # α = @. (ipx+π/2) #- acos(vx1./sqrt(vx1^2 + vy1^2))
    # idx = α .>2π
    # α[idx] .-= 2π
    ## ROTATE VISCOUS TENSOR ===========================================
    R1 = [@SMatrix [
        vx1[i] 0 vx2[i]
        0 1 0
        vy1[i] 0 vy2[i]
    ] for i in CartesianIndices(vx1)]
    C1rot = [directRotation3D(R1[i], C[i]) for i in CartesianIndices(vx1)]

    R2 = [
        @SMatrix [
            cos(ipx[i]) 0 sin(ipx[i])
            0 1 0
            -sin(ipx[i]) 0 cos(ipx[i])
        ] for i in CartesianIndices(vx1)
    ]
    Crot = [directRotation3D(inv(R2[i]), C1rot[i]) for i in CartesianIndices(vx1)]
    return Crot
end

function directRotation3D(R, C)
    ## Tensor coefficients
    a = C[1, 1]
    b = C[2, 2]
    c = C[3, 3]
    d = C[4, 4]
    e = C[5, 5]
    f = C[6, 6]
    g = C[1, 2]
    h = C[1, 3]
    l = C[2, 3]
    ## Rotation matrix
    x1 = R[1, 1]
    x2 = R[1, 2]
    x3 = R[1, 3]
    y1 = R[2, 1]
    y2 = R[2, 2]
    y3 = R[2, 3]
    z1 = R[3, 1]
    z2 = R[3, 2]
    z3 = R[3, 3]
    ## Rotated components
    n11 =
        4 * ((e * x3^2 + f * x2^2) * x1^2 + d * x2^2 * x3^2) +
        (h * x1^2 + l * x2^2 + c * x3^2) * x3^2 +
        (g * x2^2 + h * x3^2 + a * x1^2) * x1^2 +
        (g * x1^2 + l * x3^2 + b * x2^2) * x2^2
    n12 =
        4 * ((e * x3 * y3 + f * x2 * y2) * x1 * y1 + d * x2 * x3 * y2 * y3) +
        (h * x1^2 + l * x2^2 + c * x3^2) * y3^2 +
        (g * x2^2 + h * x3^2 + a * x1^2) * y1^2 +
        (g * x1^2 + l * x3^2 + b * x2^2) * y2^2
    n13 =
        4 * ((e * x3 * z3 + f * x2 * z2) * x1 * z1 + d * x2 * x3 * z2 * z3) +
        (h * x1^2 + l * x2^2 + c * x3^2) * z3^2 +
        (g * x2^2 + h * x3^2 + a * x1^2) * z1^2 +
        (g * x1^2 + l * x3^2 + b * x2^2) * z2^2
    n14 =
        2 * (
            ((x2 * z3 + x3 * z2) * e * x1 + (y2 * z3 + y3 * z2) * d * x2) * x3 +
            (x2 * y3 + x3 * y2) * f * x1 * x2
        ) +
        (h * x1^2 + l * x2^2 + c * x3^2) * y3 * z3 +
        (g * x2^2 + h * x3^2 + a * x1^2) * y1 * z1 +
        (g * x1^2 + l * x3^2 + b * x2^2) * y2 * z2
    n15 =
        2 * (
            ((x1 * z3 + x3 * z1) * e * x1 + (y1 * z3 + y3 * z1) * d * x2) * x3 +
            (x1 * y3 + x3 * y1) * f * x1 * x2
        ) +
        (h * x1^2 + l * x2^2 + c * x3^2) * x3 * z3 +
        (g * x2^2 + h * x3^2 + a * x1^2) * x1 * z1 +
        (g * x1^2 + l * x3^2 + b * x2^2) * x2 * z2
    n16 =
        2 * (
            ((x1 * z2 + x2 * z1) * e * x1 + (y1 * z2 + y2 * z1) * d * x2) * x3 +
            (x1 * y2 + x2 * y1) * f * x1 * x2
        ) +
        (h * x1^2 + l * x2^2 + c * x3^2) * x3 * y3 +
        (g * x2^2 + h * x3^2 + a * x1^2) * x1 * y1 +
        (g * x1^2 + l * x3^2 + b * x2^2) * x2 * y2
    n22 =
        4 * ((e * y3^2 + f * y2^2) * y1^2 + d * y2^2 * y3^2) +
        (h * y1^2 + l * y2^2 + c * y3^2) * y3^2 +
        (g * y2^2 + h * y3^2 + a * y1^2) * y1^2 +
        (g * y1^2 + l * y3^2 + b * y2^2) * y2^2
    n23 =
        4 * ((e * y3 * z3 + f * y2 * z2) * y1 * z1 + d * y2 * y3 * z2 * z3) +
        (h * y1^2 + l * y2^2 + c * y3^2) * z3^2 +
        (g * y2^2 + h * y3^2 + a * y1^2) * z1^2 +
        (g * y1^2 + l * y3^2 + b * y2^2) * z2^2
    n24 =
        2 * (
            ((x2 * z3 + x3 * z2) * e * y1 + (y2 * z3 + y3 * z2) * d * y2) * y3 +
            (x2 * y3 + x3 * y2) * f * y1 * y2
        ) +
        (h * y1^2 + l * y2^2 + c * y3^2) * y3 * z3 +
        (g * y2^2 + h * y3^2 + a * y1^2) * y1 * z1 +
        (g * y1^2 + l * y3^2 + b * y2^2) * y2 * z2
    n25 =
        2 * (
            ((x1 * z3 + x3 * z1) * e * y1 + (y1 * z3 + y3 * z1) * d * y2) * y3 +
            (x1 * y3 + x3 * y1) * f * y1 * y2
        ) +
        (h * y1^2 + l * y2^2 + c * y3^2) * x3 * z3 +
        (g * y2^2 + h * y3^2 + a * y1^2) * x1 * z1 +
        (g * y1^2 + l * y3^2 + b * y2^2) * x2 * z2
    n26 =
        2 * (
            ((x1 * z2 + x2 * z1) * e * y1 + (y1 * z2 + y2 * z1) * d * y2) * y3 +
            (x1 * y2 + x2 * y1) * f * y1 * y2
        ) +
        (h * y1^2 + l * y2^2 + c * y3^2) * x3 * y3 +
        (g * y2^2 + h * y3^2 + a * y1^2) * x1 * y1 +
        (g * y1^2 + l * y3^2 + b * y2^2) * x2 * y2
    n33 =
        4 * ((e * z3^2 + f * z2^2) * z1^2 + d * z2^2 * z3^2) +
        (h * z1^2 + l * z2^2 + c * z3^2) * z3^2 +
        (g * z2^2 + h * z3^2 + a * z1^2) * z1^2 +
        (g * z1^2 + l * z3^2 + b * z2^2) * z2^2
    n34 =
        2 * (
            ((x2 * z3 + x3 * z2) * e * z1 + (y2 * z3 + y3 * z2) * d * z2) * z3 +
            (x2 * y3 + x3 * y2) * f * z1 * z2
        ) +
        (h * z1^2 + l * z2^2 + c * z3^2) * y3 * z3 +
        (g * z2^2 + h * z3^2 + a * z1^2) * y1 * z1 +
        (g * z1^2 + l * z3^2 + b * z2^2) * y2 * z2
    n35 =
        2 * (
            ((x1 * z3 + x3 * z1) * e * z1 + (y1 * z3 + y3 * z1) * d * z2) * z3 +
            (x1 * y3 + x3 * y1) * f * z1 * z2
        ) +
        (h * z1^2 + l * z2^2 + c * z3^2) * x3 * z3 +
        (g * z2^2 + h * z3^2 + a * z1^2) * x1 * z1 +
        (g * z1^2 + l * z3^2 + b * z2^2) * x2 * z2
    n36 =
        2 * (
            ((x1 * z2 + x2 * z1) * e * z1 + (y1 * z2 + y2 * z1) * d * z2) * z3 +
            (x1 * y2 + x2 * y1) * f * z1 * z2
        ) +
        (h * z1^2 + l * z2^2 + c * z3^2) * x3 * y3 +
        (g * z2^2 + h * z3^2 + a * z1^2) * x1 * y1 +
        (g * z1^2 + l * z3^2 + b * z2^2) * x2 * y2
    n44 =
        (x2 * z3 + x3 * z2)^2 * e +
        (y2 * z3 + y3 * z2)^2 * d +
        (x2 * y3 + x3 * y2)^2 * f +
        (h * y1 * z1 + l * y2 * z2 + c * y3 * z3) * y3 * z3 +
        (g * y2 * z2 + h * y3 * z3 + a * y1 * z1) * y1 * z1 +
        (g * y1 * z1 + l * y3 * z3 + b * y2 * z2) * y2 * z2
    n45 =
        (x1 * z3 + x3 * z1) * (x2 * z3 + x3 * z2) * e +
        (y1 * z3 + y3 * z1) * (y2 * z3 + y3 * z2) * d +
        (x1 * y3 + x3 * y1) * (x2 * y3 + x3 * y2) * f +
        (h * y1 * z1 + l * y2 * z2 + c * y3 * z3) * x3 * z3 +
        (g * y2 * z2 + h * y3 * z3 + a * y1 * z1) * x1 * z1 +
        (g * y1 * z1 + l * y3 * z3 + b * y2 * z2) * x2 * z2
    n46 =
        (x1 * z2 + x2 * z1) * (x2 * z3 + x3 * z2) * e +
        (y1 * z2 + y2 * z1) * (y2 * z3 + y3 * z2) * d +
        (x1 * y2 + x2 * y1) * (x2 * y3 + x3 * y2) * f +
        (h * y1 * z1 + l * y2 * z2 + c * y3 * z3) * x3 * y3 +
        (g * y2 * z2 + h * y3 * z3 + a * y1 * z1) * x1 * y1 +
        (g * y1 * z1 + l * y3 * z3 + b * y2 * z2) * x2 * y2
    n55 =
        (x1 * z3 + x3 * z1)^2 * e +
        (y1 * z3 + y3 * z1)^2 * d +
        (x1 * y3 + x3 * y1)^2 * f +
        (h * x1 * z1 + l * x2 * z2 + c * x3 * z3) * x3 * z3 +
        (g * x2 * z2 + h * x3 * z3 + a * x1 * z1) * x1 * z1 +
        (g * x1 * z1 + l * x3 * z3 + b * x2 * z2) * x2 * z2
    n56 =
        (x1 * z2 + x2 * z1) * (x1 * z3 + x3 * z1) * e +
        (y1 * z2 + y2 * z1) * (y1 * z3 + y3 * z1) * d +
        (x1 * y2 + x2 * y1) * (x1 * y3 + x3 * y1) * f +
        (h * x1 * z1 + l * x2 * z2 + c * x3 * z3) * x3 * y3 +
        (g * x2 * z2 + h * x3 * z3 + a * x1 * z1) * x1 * y1 +
        (g * x1 * z1 + l * x3 * z3 + b * x2 * z2) * x2 * y2
    n66 =
        (x1 * z2 + x2 * z1)^2 * e +
        (y1 * z2 + y2 * z1)^2 * d +
        (x1 * y2 + x2 * y1)^2 * f +
        (h * x1 * y1 + l * x2 * y2 + c * x3 * y3) * x3 * y3 +
        (g * x2 * y2 + h * x3 * y3 + a * x1 * y1) * x1 * y1 +
        (g * x1 * y1 + l * x3 * y3 + b * x2 * y2) * x2 * y2

    return @SMatrix [
        n11 n12 n13 n14 n15 n16
        n12 n22 n23 n24 n25 n26
        n13 n23 n33 n34 n35 n36
        n14 n24 n34 n44 n45 n46
        n15 n25 n35 n45 n55 n56
        n16 n26 n36 n46 n56 n66
    ]
end

######
# distance(p1::Point2D{Polar},p2::Point2D{Polar}) = sqrt( p1.z^2 + p2.z^2 - 2p1.z*p2.z*cos(p1.x-p2.x))
# distance(p1::Point2D{Polar},p2::Array{Point2D{Polar},N}) where {N} = 
#     [sqrt( p1.z^2 + p2[u].z^2 - 2p1.z*p2[u].z*cos(p1.x-p2[u].x)) for u in CartesianIndices(p2)]

# distance(p1::Point2D{Cartesian},p2::Point2D{T}) where {T} = sqrt((p1.x-p2.x)^2 + (p1.z-p2.z)^2)
# distance(p1::Point2D{Cartesian},p2::Array{Point2D{Polar},N}) where {N} = 
#     [sqrt((p1.x-p2[u].x)^2 + (p1.z-p2[u].z)^2) for u in CartesianIndices(p2)]

# fname = "/home/albert/Documents/JuM2TRI/JuM2TRI/JuliaFEM/output/IsoviscousAnisotropic_1e4_fullTensor/file_54.h5"
# # fname = "/home/albert/Documents/JuM2TRI/JuM2TRI/JuliaFEM/output/Anisotropic1e6_RK4/file_42.h5"
# M, V, U, 𝓒, Fxx, Fzz, Fxz, Fzx, a1, a2, vx1, vx2, vy1, vy2 = loader(fname);

# # fname = "/home/albert/Documents/JuM2TRI/JuM2TRI/JuliaFEM/output/Isotropic1e6_RK4/file_42.h5"
# # M, Viso, Uiso, 𝓒, Fxx, Fzz, Fxz, Fzx, a1, a2, vx1, vx2, vy1, vy2 = loader(fname,false);
# # # FSE = [FiniteStrainEllipsoid{Float64}(vx1[u], vx2[u], vy1[u], vy2[u], a1[u], a2[u]) for u in CartesianIndices(vx1)]

# # x, z = M.x, M.z
# # θ, r = M.θ, M.r

# cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE, M.e2n[i,1:3]) for i in axes(M.e2n,1)]
# vtkfile = vtk_grid("my_vtk_file", M.x, M.z, cells)    # 2D
# vtkfile["color", VTKCellData()] = els_colors
# vtkfile["Temperature", VTKPointData()] = V.T
# vtkfile["Vel", VTKPointData()] = (U.x,U.z)
# outfiles = vtk_save(vtkfile)

# # pvd = paraview_collection("my_pvd_file")
# # pvd[1] = vtkfile
# # vtk_save(pvd)

# # ## Plot
# xz = [M.x M.z]
# ax,pl=mesh(xz,M.e2n,color = U.r, colormap = ColorSchemes.balance,shading = false) 
# # x,z=polar2cartesian(ipx,ipz)

# fig = Figure(resolution = (1600, 900), backgroundcolor = RGBf0(0.98, 0.98, 0.98))

# ax1 = fig[1, 1] = Axis(fig, title = "Anisotropic")
# mesh!(ax1, xz,M.e2n,color = V.T, colormap = ColorSchemes.balance,shading = false) 
# arrows!(ax1,vec(x),vec(z),vec(vx1)/1e2, vec(vy1)./1e2, arrowsize = 1e-5)
# xlims!(ax1,(-2.12,0))
# ylims!(ax1,(0,2.12))

# ax2 = fig[1, 2] = Axis(fig, title = "Isotropic")
# mesh!(ax2, xz,M.e2n,color = Viso.T, colormap = ColorSchemes.balance,shading = false) 
# arrows!(ax2,vec(x),vec(z),vec(vx1)/1e2, vec(vy1)./1e2, arrowsize = 1e-5)
# xlims!(ax2,(0,2.12))
# ylims!(ax2,(0,2.12))

# hideydecorations!(ax2,grid=false)
# linkaxes!(ax1,ax2)
# fig

# ## =========================

# ax,pl=mesh(xz,M.e2n,color = V.T, colormap = ColorSchemes.balance,shading = false) 

# cm = colorlegend(
#     pl[end],             # access the plot of Scene p1
#     raw = true,          # without axes or grid
#     camera = campixel!,  # gives a concrete bounding box in pixels
#                          # so that the `vbox` gives you the right size
#     width = (            # make the colorlegend longer so it looks nicer
#         30,              # the width
#         400              # the height
#     )
#     )

# scene_final = vbox(pl, cm) # put the colorlegend and the plot together in a `vbox`

# xlims!((0,2.12))
# ylims!(0,2.12)

# ipx = [IntC[i,j].x for i in axes(IntC,1), j in 1:6]
# ipz = [IntC[i,j].z for i in axes(IntC,1), j in 1:6]
# η11 = 𝓒.η11[:,1:6]
# η33 = 𝓒.η33[:,1:6]
# η55 = 𝓒.η55[:,1:6]
# η13 = 𝓒.η13[:,1:6]

# # x,z=polar2cartesian(ipx,ipz)
# # arrows!(vec(x),vec(z),vec(vx1)/1e2, vec(vy1)./1e2, arrowsize = 1e-5)
# pl=scatter(vec(x),vec(z),color=(vec(η11)),colormap = ColorSchemes.balance,markersize=0.1)

# cm = colorlegend(
#     pl[end],             # access the plot of Scene p1
#     raw = true,          # without axes or grid
#     camera = campixel!,  # gives a concrete bounding box in pixels
#                          # so that the `vbox` gives you the right size
#     width = (            # make the colorlegend longer so it looks nicer
#         30,              # the width
#         400              # the height
#     )
#     )
# scene_final = vbox(pl, cm) # put the colorlegend and the plot together in a `vbox`

# x = LinRange(0,3pi,200); y = sin.(x)
# lin = lines(x, y, padding = (0.0, 0.0), axis = (
#     names = (axisnames = ("", ""),),
#     grid = (linewidth = (0, 0),),
# ))

# ##
# Q1,Q2,Q3,Q4 = quadrants(M.θ, M.r, M.e2n)

# mesh(xz, Q1, color = V.T,
#         colormap = ColorSchemes.Accent_6,shading = true)

# mesh(xz, Q3, color = log10.(V.η),
#         colormap = ColorSchemes.balance,shading = false)
