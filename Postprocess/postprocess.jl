include("Plotting/misc.jl")

function loadmesh(fname)
    fid = h5open(fname,"r")
    
    # -- Load mesh
    M = fid["MESH"]
    θ = read(M,"x")
    r = read(M,"z")
    e2n = read(M,"EL2NOD")
    x, z = polar2cartesian(θ, r) 
    M = Mesh(x,z,θ,r, Array(e2n'))

end

function Urms(M,p)
      # -- output files
      files = readdir(p)
      # -- allocations
      rms = Vector{Float64}(undef,length(files))
      t = similar(rms)
      A = π*(2.22^2-1.22^2)
      Ael =  element_area(M.x, M.z, M.e2n')
      # -- Calculate Urms and Nusselt number
      @inbounds for (i, file) in enumerate(files)
        _,V,U,𝓒, Fxx, Fzz, Fxz, Fzx, a1, a2, vx1, vx2, vy1, vy2,  t[i] = loader(joinpath(p, files[i]), isotropic=true)
        # remove net translations
        # a,b = extrema(U.x)
        # c = (a+b)*0.5
        # U.x .-= c
        # a,b = extrema(U.z)
        # c = (a+b)*0.5
        # U.z .-= c
        # rms
        # vel = @. @views ((U.x[M.e2n[:, 1:3]]/M.r[M.e2n[:, 1:3]])^2+U.z[M.e2n[:, 1:3]]^2)
        vel = @. @views ((U.x[M.e2n[:, 1:3]])^2+U.z[M.e2n[:, 1:3]]^2)
        Atop = mean(vel, dims=2).*Ael
        rms[i] = sqrt(sum(Atop) /A)
      end
      return rms, t
end

function fem_urms(M, p, θ3, r3)
    # -- output files
    files = readdir(p)
    # -- allocations
    rms = Vector{Float64}(undef,length(files))
    t = similar(rms)
    A = π*(2.22^2-1.22^2)
    # -- Calculate Urms and Nusselt number
    @inbounds for (i, file) in enumerate(files)
       _,V,U,𝓒, Fxx, Fzz, Fxz, Fzx, a1, a2, vx1, vx2, vy1, vy2,  t[i] = loader(joinpath(p,files[i]), isotropic=true)
      #  vel = @. (U.θ^2+U.r^2)
       vel = @. (U.θ^2+U.r^2)
       intV = volume_integral(vel, M.e2n', θ3, r3)
       A = volume_integral(ones(size(vel)), M.e2n', θ3, r3)
       rms[i] = sqrt(intV)/sqrt(A)
    end
    return rms, t
end

function fem_NU(M, p)
  # -- output files
  files = readdir(p)
  # -- allocations
  Nu_top = Vector{Float64}(undef,length(files))
  Nu_bot = similar(Nu_top)
  t = similar(Nu_top)
  r_in, r_out = 1.22, 2.22
  A = π*(r_out^2-r_in^2)

  θ, r = M.θ, M.r
  top, bottom = extrema(r)
  itop, ibottom = findall(x->x==top, r), findall(x->x==bottom, r)

  runique = unique(r); sort!(runique)
  itop_minus, ibot_minus = findall(x->x==runique[end-1], r), findall(x->x==runique[2], r)
  θtop, θbottom = view(θ, itop_minus), view(θ, itop_minus)
  isort = sortperm(θtop)
  θtop_minus, θbot_minus = θ[itop_minus], θ[ibot_minus]

  Δr = runique[2] - runique[1]
  Δθ = θtop[1] - θtop[2]
  f = r_in/r_out
  A = log(f)/(2π*r_out*(1-f))

  # -- Nusselt number
  @inbounds for (i, file) in enumerate(files)
     _,V,U,𝓒, Fxx, Fzz, Fxz, Fzx, a1, a2, vx1, vx2, vy1, vy2,  t[i] = loader(joinpath(p,files[i]), isotropic=true)

     Ttop_minus, Tbot_minus = V.T[itop_minus], V.T[ibot_minus]
     dTdr_top = @. (0.0 - Ttop_minus)/Δr
     dTdr_bot = @. (Tbot_minus - 1.0)/Δr
   
     Nu_top[i] = A * Δθ * sum(dTdr_top)
     Nu_bot[i] = A * Δθ * sum(dTdr_bot)

  end
  
  return Nu_top, Nu_bot, t
end

src = joinpath(pwd(), "output")
fldr = "/IsoviscousAnisotropic_1e4_benchmark_4_plumes_GC_longrun_rsquared/"

fname = "file_1.h5"
M = loadmesh(string(src, fldr, fname))
p = string(src, fldr)

xz = hcat(M.x, M.z)
e2n = M.e2n'
θ3, r3 = M.θ[e2n[1:3,:]], M.r[e2n[1:3,:]]
fixangles!(θ3)
# rms2, t2= single_urms(M2, p2);

p = string(src, fldr)
rms_fem, _ = fem_urms(M, p, θ3, r3)
rms1, t1 = Urms(M, p)
perm = sortperm(t1)

s1=scatter(t1, rms1, color=:red)
scatter!(t1, rms_fem, color=:blue)

fldr = "/IsoviscousAnisotropic_1e3_benchmark_4_plumes_RK42/"
rms_fem, _ = fem_urms(M, p, θ3, r3)
rms2, t2 = Urms(M,p)

scatter(t1, rms1, color=:red)
scatter!(t1, rms_fem, color=:blue)

##
ipart = 5622
t = [particle_info[i].t for i in axes(particle_info,1)]
iel = t[ipart]
iparts = findall(x->x==iel,t)
e2n = EL2NOD_P1[:,iel]

xp = [particle_info[i].CCart.x for i in iparts]
zp = [particle_info[i].CCart.z for i in iparts]
xe = x[e2n]
ze = z[e2n]
Tp = particle_fields.T[iparts]
Te = T[e2n]
T1e = T1[e2n]
T2e = T2[e2n]

scatter(xp,zp,Tp,color=:blue, markersize= 0.5)
scatter!(xe,ze,Te,color=:red, markersize= 0.5)
scatter!(xe,ze,T1e,color=:orange, markersize= 0.5)
scatter!(xe,ze,T2e,color=:green, markersize= 0.5)

#####################################

t0 = [particle_info0[i].CCart for i in 1:10]
t1 = [particle_info[i].CCart for i in 1:10]

t0_x = [t0[i].x for i in 1:10] 
t0_z = [t0[i].z for i in 1:10] 

t1_x = [t1[i].x for i in 1:10] 
t1_z = [t1[i].z for i in 1:10] 

scatter(t0_x, t0_z)
scatter!(t1_x, t1_z, color=:red)

d(a1, a2, b1, b2) = @. √( (a1-b1)^2 + (a2-b2)^2 )

dist = d(t0_x, t0_z, t1_x, t1_z)