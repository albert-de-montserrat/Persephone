using Makie
using ColorSchemes

XY = Matrix(xy')
faces = Matrix(EL2NOD[1:6, :]')
xx, zz = XY[:, 1], XY[:, 2]

Uth = U[1:2:end]
Ur = U[2:2:end]

a1 = [FSE[i].a1 for i in axes(FSE, 1)]

Ux = [Ucartesian[i].x for i in axes(Ucartesian, 1)]
Uz = [Ucartesian[i].z for i in axes(Ucartesian, 1)]
Fxx = [F[i][1, 1] for i in CartesianIndices(F)]

x, z = polar2cartesian(ipx, ipz)
scatter(vec(x), vec(z); color=vec(Fxx), markersize=1, colormap=:Spectral)

fig = Figure(; resolution=(1200, 900), backgroundcolor=RGBf0(0.98, 0.98, 0.98))
ax1 = fig[1, 1] = Axis(fig; title="t = $Time")
m1 = mesh!(ax1, XY, faces; color=sCG[1:64000], colormap=:Spectral, shading=false)

ax2 = fig[1, 2] = Axis(fig; title="t = $Time")
m2 = mesh!(ax2, XY, faces; color=sPenalty[1:64000], colormap=:Spectral, shading=false)

ax1.aspect = DataAspect()
ax2.aspect = DataAspect()
linkaxes!(ax1, ax2)
# pl= mesh(XY,faces,color = log10.(η),
# colormap = ColorSchemes.magma.colors,shading = false) 

cm = colorlegend(
    pl[end];             # access the plot of Scene p1
    raw=true,          # without axes or grid
    camera=campixel!,  # gives a concrete bounding box in pixels
    # so that the `vbox` gives you the right size
    width=(            # make the colorlegend longer so it looks nicer
        30,              # the width
        540,              # the height
    ),
)

scene_final = vbox(pl, cm) # put the colorlegend and the plot together in a `vbox`

x = [particle_info[i].CCart.x for i in axes(particle_info, 1)]
z = [particle_info[i].CCart.z for i in axes(particle_info, 1)]
θ = [particle_info[i].CPolar.x for i in axes(particle_info, 1)]

uxp = [particle_info[i].UCart.x for i in axes(particle_info, 1)]
uzp = [particle_info[i].UCart.z for i in axes(particle_info, 1)]

scatter(x, z; color=particle_fields.T)

scatter(T, XY[:, 2])

scatter(uxp, z)

# Polar plots
ilost = unique(t_lost)
id = ilost[2]
xp = particle_info[id].CPolar.x
zp = particle_info[id].CPolar.z
vertices = [Point2D{Polar}(rand(), rand()) for i in 1:3] # element vertices 

previous_el = particle_info[id].t
getvertices!(vertices, θThermal, rThermal, previous_el)
xt = [vertices[i].x for i in 1:3]
zt = [vertices[i].z for i in 1:3]

Plots.scatter(xt, zt; marker=:hex, color=:red, label=false)
Plots.scatter!([xp], [zp]; color=:blue, label=false)
Plots.xlabel!("th")
Plots.ylabel!("r")

ineigh = Int.(neighbours0[previous_el])
# centroiddistance!(dist2centroid,imin,ineigh,particle,CC)
dist2centroid = centroiddistance(ineigh, particle, CentC)
imin = argmin(dist2centroid)

# -- Calculate barycentric coordinates and check if point is inside triangle
id0 = ineigh[imin]
getvertices!(vertices, θThermal, rThermal, id0)
xt = [vertices[i].x for i in 1:3]
zt = [vertices[i].z for i in 1:3]

Plots.scatter!(xt, zt; color=:green)

for k in ineigh
    getvertices!(vertices, θ3, r3, k)
    xt = vcat(xt, [vertices[i].x for i in 1:3])
    zt = vcat(zt, [vertices[i].z for i in 1:3])
end
Plots.scatter!(xt, zt)

# ==========================
# Cartesian plots
# Polar plots
ilost = unique(t_lost)
id = ilost[2]
xp = particle_info[id].CCart.x
zp = particle_info[id].CCart.z
vertices = [Point2D{Cartesian}(rand(), rand()) for i in 1:3] # element vertices 

xx = xy[1, :]
zz = xy[2, :]
x3 = xx[EL2NOD_P1]
z3 = zz[EL2NOD_P1]

previous_el = particle_info[id].t
getvertices!(vertices, x3, z3, previous_el)
xt = [vertices[i].x for i in 1:3]
zt = [vertices[i].z for i in 1:3]

Plots.scatter(xt, zt; marker=:hex, color=:red, label=false)
Plots.scatter!([xp], [zp]; color=:blue, label=false)
Plots.xlabel!("th")
Plots.ylabel!("r")

ineigh = Int.(neighbours0[previous_el])
# centroiddistance!(dist2centroid,imin,ineigh,particle,CC)
dist2centroid = centroiddistance(ineigh, particle, CentC)
imin = argmin(dist2centroid)

# -- Calculate barycentric coordinates and check if point is inside triangle
id0 = ineigh[imin]
getvertices!(vertices, x3, z3, id0)
xt = [vertices[i].x for i in 1:3]
zt = [vertices[i].z for i in 1:3]

Plots.scatter!(xt, zt; color=:green)

for k in ineigh
    getvertices!(vertices, x3, z3, k)
    xt = vcat(xt, [vertices[i].x for i in 1:3])
    zt = vcat(zt, [vertices[i].z for i in 1:3])
end
Plots.scatter!(xt, zt; color=:orange)

## 
xip = [IntC[i, j].x for i in axes(IntC, 1), j in 1:6]
zip = [IntC[i, j].z for i in axes(IntC, 1), j in 1:6]
η11 = 𝓒.η11[:, 1:6]
η55 = 𝓒.η55[:, 1:6]
η33 = 𝓒.η33[:, 1:6]
η13 = 𝓒.η13[:, 1:6]

x, z = polar2cartesian(ipx, ipz)
pl = scatter(vec(x), vec(z); color=vec(a1))

cm = colorlegend(
    pl[end];             # access the plot of Scene p1
    raw=true,          # without axes or grid
    camera=campixel!,  # gives a concrete bounding box in pixels
    # so that the `vbox` gives you the right size
    width=(            # make the colorlegend longer so it looks nicer
        30,              # the width
        540,              # the height
    ),
)

scene_final = vbox(pl, cm) # put the colorlegend and the plot together in a `vbox`
Fxx = [F[i][1, 1] for i in eachindex(F)]

vx1 = [FSE[i].x1 for i in eachindex(FSE)]
vy1 = [FSE[i].y1 for i in eachindex(FSE)]
vx10 = [FSE0[i].x1 for i in eachindex(FSE)]
vy10 = [FSE0[i].y1 for i in eachindex(FSE)]

a1 = [FSE[i].a1 for i in eachindex(FSE)]
a2 = [FSE[i].a2 for i in eachindex(FSE)]
a10 = [FSE0[i].a1 for i in eachindex(FSE)]
a20 = [FSE0[i].a2 for i in eachindex(FSE)]

arrows(vec(x), vec(z), vec(vx1) * 1e-2, vec(vy1) * 1e-2; arrowsize=1e-7)
x, z = polar2cartesian(ipx, ipz)
arrows!(vec(x), vec(z), vec(vx1) * 1e-2, vec(vy1) * 1e-2; arrowsize=1e-7)
arrows!(vec(x), vec(z), vec(vx10) * 1e-2, vec(vy10) * 1e-2; arrowsize=1e-7, color=:red)
