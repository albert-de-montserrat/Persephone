include("Mixer.jl")
include(joinpath(pwd(), "Postprocess/plot_fields.jl"))
####
        fnumber = 100
        file = string("file_", 135, ".h5")
        shared_path = "/home/albert/Documents/JuM2TRI/NoAvx/output/"
        fldrs = [
                "FSE_test" # TOP RIGHT QUADRANT
                "FSE_test_notfull" # TOP LEFT QUADRANT
                "FSE_test_isolitho" # BOT RIGHT QUADRANT
                "FSE_test_aniconstant" # BOT LEFT QUADRANT
        ]
        fname = [string(shared_path, folder, "/", file) for folder in fldrs]
        plot_4velocities(gr, fname)
####
path = "/home/albert/Desktop/OUTPUT/Persephone/Urms/"
fn1 = string(path,
               "Isotropic_1e5/",
               "file_53.h5")

fn1 = string(path,
               "Isotropic_1e4/",
               "file_108.h5")
               
path = "/home/albert/Documents/JuM2TRI/OUTPUT/"
fstats = string(path,
   "Isotropic_1e5/",
   "statistics.txt")
               
plot_stats(fstats)

# plot_anisotropy(fname, gr, isave = "no")

# fname = string(path,
#                "Tdep_iso_1e6/",
#                "file_238.h5")

path = "/home/albert/Documents/JuM2TRI/NoAvx/output/"

path = "/home/albert/Documents/JuM2TRI/OUTPUT/"

fanimation = string(path, "Tdeep")
animation_node_black(gr, fanimation)

fname = string(path, "Tdep_Isotropic_1e5_aspect/",
                "file_143.h5")

M, V, U, 𝓒, Fxx, Fzz, Fxz, Fzx, a1, a2, vx1, vx2, vy1, vy2 = loader(fname, isotropic=true);

figT = plot_node_black(V.T, gr)

fname = string("/home/albert/Desktop/output/test/",
        "file_161.h5")

M, V, U, 𝓒, Fxx, Fzz, Fxz, Fzx, a1, a2, vx1, vx2, vy1, vy2 = loader(fname, isotropic=true);
# Ael = element_area(gr)

figT = plot_node_black(V.T, gr)

figT = plot_node(U.θ, gr)
figT = plot_node(V.T, gr)

Fxxn = ip2node(gr.e2n, Fxx)
Fzzn = ip2node(gr.e2n, Fzz)
Fxzn = ip2node(gr.e2n, Fxz)
Fzxn = ip2node(gr.e2n, Fzx)

f = plot_node_black(Fxxn, gr)
f = plot_node_black(Fzzn, gr)
f = plot_node_black(Fxzn, gr)
f = plot_node_black(Fzxn, gr)

a1n = ip2node(gr.e2n, Ael, a1)
f = plot_node(a1n, gr)
figT = plot_node_black((a1n), gr)

a2n = ip2node(gr.e2n, Ael, a2)
a2n = max.(0.0, a2n)
f = plot_node(a2n, gr)
figT = plot_node_black(log10.(a2n), gr)

f = plot_node(log10.(a1n./a2n), gr)
figT = plot_node_black(log10.(a1n./a2n), gr)

# Grid to resample
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
xg = range(0, 2π, length = 96)
yg = range(1.22, 2.22, length = 9)
X = xg'.*ones(length(yg))
Y = reverse(yg).*ones(1,length(xg))
xc, yc = polar2cartesian(X,Y)

F = [@SMatrix [Fxx[i] Fxz[i]; Fzx[i] Fzz[i]] for i in CartesianIndices(Fxx)]
# FSE = computeFSE(F)
FSE = getFSE(F, FSE)

Fx1 = [FSE[i].x1 for i in CartesianIndices(FSE)]
Fy1 = [FSE[i].y1 for i in CartesianIndices(FSE)]
a1 = [FSE[i].a1 for i in CartesianIndices(FSE)]
a2 = [FSE[i].a2 for i in CartesianIndices(FSE)]

# α = atand.(Fy1, Fx1)
# αn = ip2node(gr.e2n, Ael, α)
# figT = plot_node((αn), gr)

# Resample eigenvectors and velocity onto structured grid
vx1g = togrid(xg, yg, vec(ipx), vec(ipz), vec(Fx1))
vy1g = togrid(xg, yg, vec(ipx), vec(ipz), vec(Fy1))
vx2g = togrid(xg, yg, vec(ipx), vec(ipz), vec(Fx2))
vy2g = togrid(xg, yg, vec(ipx), vec(ipz), vec(Fy2))

# Resample eigenvalues onto structured grid
a1g = togrid(xg, yg, vec(ipx), vec(ipz), vec(a1))
a1g ./= maximum(a1g)

n = length(vx1g)
scaling = 2.0./a1g
scaling = 10*a1g./a1g
points = [Point2f0((xc[i]-vx1g[i]/2/scaling[i]), yc[i]-vy1g[i]/2/scaling[i]) => 
          Point2f0((xc[i]+vx1g[i]/2/scaling[i]), yc[i]+vy1g[i]/2/scaling[i]) 
          for i in eachindex(vx1g)]

linesegments(points, color = :red, linewidth = 2)

scaling = 5
arrows(vec(xc), vec(yc), 
        vec(vx1g)/scaling/2, vec(vy1g)/scaling/2,
        arrowsize = 5e-3, arrowcolor = :green, linecolor = :green, linewidth =2)

arrows!(vec(xc), vec(yc), 
        -vec(vx2g)/scaling/2, -vec(vy2g)/scaling/2,
        arrowsize = 1e-10, arrowcolor = :blue, linecolor = :blue, linewidth =2) 
        
scaling = 1e3
arrows!(gr.x, gr.z,
        U.x/scaling, U.z/scaling,
        arrowsize = 5e-3, arrowcolor = :red, linecolor = :red, linewidth =2)
        
scaling = 1e1
arrows!(vec(ix), vec(iz), 
        vec(Fx1)/scaling/2, vec(Fy1)/scaling/2,
        arrowsize = 1e-3, arrowcolor = :blue, linecolor = :blue, linewidth =2)
###############################################################################################

idx = findall((0.56 .≤ gr.x .≤ 0.58) .& (1.125 .≤ gr.z .≤ 1.150))
idx = findall((-0.05 .≤ gr.x .≤ 0.0) .& (1.2 .≤ gr.z .≤ 1.25))

U = findall(gr.e2n .== idx[end])
iel = U[1].I[2]
xel = gr.x[gr.e2n[:,iel]]
zel = gr.z[gr.e2n[:,iel]]

𝓒.η35[iel, 1:6]
𝓒.η15[iel, 1:6]

a = α[iel, 1]
a2[iel, :]
a1[iel, :]

FSE = [FiniteStrainEllipsoid(FSE[i].x1, FSE[i].x2, FSE[i].y1, FSE[i].y2, 5.0, 0.9) for i in CartesianIndices(FSE)]

uxip,uzip = velocities2ip(Ucartesian,gr.e2n)
FSE = [FiniteStrainEllipsoid(uxip[i], uzip[i], uzip[i], FSE[i].y2, 1.2, 0.9) for i in CartesianIndices(FSE)]


####################
total = 3.13
Stokes = 1.94

# Section                         ncalls     time   %tot     alloc   %tot
#  ───────────────────────────────────────────────────────────────────────
# Stokes                               1    1.15s  35.9%   1.10GiB  54.9%
# PCG solver                         1    938ms  29.3%    552MiB  26.9%
# Assembly                           1    105ms  3.28%    236MiB  11.5%
# BCs                                1    105ms  3.28%    335MiB  16.3%
# Remove net rotation                1   2.05ms  0.06%   3.04MiB  0.15%
# Particle advenction                  1    844ms  26.3%    359MiB  17.5%
# Advection                          1    698ms  21.8%    187MiB  9.10%
#   locate particle                  4    451ms  14.1%   6.45MiB  0.31%
#   velocity → particle              4   85.8ms  2.68%   11.4KiB  0.00%
#   particle advection               3   75.9ms  2.37%   77.2MiB  3.76%
# T → particle                       1    112ms  3.48%    147MiB  7.14%
#   step 2                           1   79.8ms  2.49%    147MiB  7.14%
#   step 1                           1   30.9ms  0.96%   3.00KiB  0.00%
#   step 3                           1    850μs  0.03%   2.75KiB  0.00%
# F → particle                       1   18.8ms  0.59%   2.81KiB  0.00%
# Thermal diffusion                    1    521ms  16.2%    221MiB  10.8%
# Solve                              1    472ms  14.7%   88.9MiB  4.33%
# Assembly                           1   43.9ms  1.37%    119MiB  5.79%
# BCs                                1   4.49ms  0.14%   13.2MiB  0.64%
# Add/reject particles                 1    313ms  9.76%   15.8MiB  0.77%
# Particles                            1    147ms  4.60%   1.61MiB  0.08%
# Locate                             1    147ms  4.59%   1.61MiB  0.08%
# Particle to node/ip                  1    123ms  3.84%    255MiB  12.4%
# T -> node                          1   71.0ms  2.21%    133MiB  6.49%
# Fij -> ip                          1   51.8ms  1.62%    122MiB  5.96%
# Get and rotate viscous tensor        1   50.0ms  1.56%   19.5MiB  0.95%
# Stress                               1   40.2ms  1.25%   27.7MiB  1.35%
# Finite Strain Ellipsoid              1   14.9ms  0.47%   18.0MiB  0.88%
# Run stats                            1   1.86ms  0.06%   7.04MiB  0.34%
# Material properties                  1   79.5μs  0.00%    516KiB  0.02%
# ───────────────────────────────────────────────────────────────────────

Stokes = [ 1.09 1.10e9]  
to_particle = [ (112+18.8)*1e-3 (147e6+2.81e3)]
advenction = [ 698e-3 187e6]
diffusion = [521e-3  221e6]
particle2node = [123e-3 255e6]
tensor = [24.9e-3 9e6]
fse = [8.42e-3 18e6]
Stress = [40.2e-3  27.7e6]

P = vcat(
        Stokes,
        to_particle,
        advenction,
        diffusion,
        particle2node,
        tensor,
        fse,
        Stress,
)

names = [
        "Stokes",
        "field to particles",
        "advenction",
        "thermal diffusion",
        "field to nodes/ip",
        "tensor",
        "FSE",
        "stress",
]

t = P[:,1]
memory = P[:,2]

itime = sortperm(t)
imem = sortperm(memory)

fig = Figure(resolution = (1000, 700))

ax1 = fig[1, 1] = Axis(fig, xticklabelrotation = deg2rad(45))

p = barplot!(ax1, t[itime], color = 1:8)

ylims!(ax1, (0.0, maximum(t)) )

ax1.ylabel = "computational time [s]"
ax1.xticks = (1:8, names[itime])

hidedecorations!(ax1, ticks=false, ticklabels=false, label=false)
fig

save("ComputationalTime.png", fig)


function mygemm!(C, A, B)
        @turbo for m ∈ axes(A,1), n ∈ axes(B,2)
            Cmn = zero(eltype(C))
            for k ∈ axes(A,2)
                Cmn += A[m,k] * B[k,n]
            end
            C[m,n] = Cmn
        end
end

function mygemm2!(C, A, B)
        rowA = Vector{eltype(C)}(undef,size(A,1))
         for m ∈ axes(A,1)
                
                #cache loop
                for k in axes(A,2)
                        @inbounds rowA[k] = A[m,k]
                end

                @turbo for n ∈ axes(B,2)
                        Cmn = zero(eltype(C))
                        for k ∈ axes(A,2)
                                Cmn += rowA[k] * B[k,n]
                        end
                        C[m,n] = Cmn
                end
        end
end

M = K = N = 100;
C1 = Matrix{Float64}(undef, M, N); A = randn(M, K); B = randn(K, N);
C2 = Matrix{Float64}(undef, M, N);
C = Matrix{Float64}(undef, M, N);

@btime mygemm!($C, $A, $B)
@btime mygemm2!($C, $A, $B)

mygemm!(C1, A, B); C1
mygemm2!(C2, A, B); C2

function matmul3x3(a::SMatrix, b::SMatrix)
        D1 = a.data; D2 = b.data
        # Extract data from matrix into SIMD.jl Vec
        SV11 = Vec((D1[1], D1[2], D1[3]))
        SV12 = Vec((D1[4], D1[5], D1[6]))
        SV13 = Vec((D1[7], D1[8], D1[9]))
    
        # Form the columns of the resulting matrix
        r1 = muladd(SV13, D2[3], muladd(SV12, D2[2], SV11 * D2[1]))
        r2 = muladd(SV13, D2[6], muladd(SV12, D2[5], SV11 * D2[4]))
        r3 = muladd(SV13, D2[9], muladd(SV12, D2[8], SV11 * D2[7]))
    
        return SMatrix{3,3}((r1[1], r1[2], r1[3],
                             r2[1], r2[2], r2[3],
                             r3[1], r3[2], r3[3]))
    end




    ###############
    Fxx_ip = [F[i][1,1] for i in CartesianIndices(F)]

    t = [particle_info[i].t_parent for i in eachindex(particle_info)]
    idx = findall(t.==1)
    xp = [particle_info[i].CCart.x for i in idx]
    zp = [particle_info[i].CCart.z for i in idx]
    Fxx_p = particle_fields.Fxx[idx]
    
    scatter(xp, zp, Fxx_p, markersize=1, color=:red)
    scatter!(ix[1,:], iz[1,:], Fxx_ip[1,:], markersize=1, color=:blue)

    scatter(ix[1,:], iz[1,:], markersize=2,color=:blue)
    scatter!(xn, zn, markersize=2)
    scatter!(xp, zp, Fxx_p, markersize=2, color=:red)