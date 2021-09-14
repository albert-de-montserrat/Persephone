
function stress(
    iel,
    KS,
    MS,
    GS,
    FS,
    DoF_U,
    DoF_P,
    θ,
    r,
    A::ScratchThermalGlobal,
    SF_Stokes::ShapeFunctionsStokes,
    g, 
    ρ, 
    η, 
    𝓒,
)

    N, ∇N, NP, dN3ds, w_ip, N3 = 
        SF_Stokes.N, SF_Stokes.∇N, SF_Stokes.NP, SF_Stokes.dN3ds, SF_Stokes.w_ip, SF_Stokes.N3 
    
    valU, valP = Val(12), Val(3)

    # Polar coordinates of element nodes
    θ_el = @SVector [θ[i, iel] for i in 1:3]
    r_el = @SVector [r[i, iel] for i in 1:3]
    coords = SMatrix{3,2}([θ_el r_el])

    # Jacobian n. 1 (p:=polar, l:=local): reference element --> current element
    J_pl = dN3ds * coords
    detJ_pl = mydet(J_pl)

    R_21 = r_el[2] - r_el[1]  # = -detJa_PL*deta_dth
    R_31 = r_el[3] - r_el[1]  # =  detJa_PL*dxi_dth
    Th_31 = θ_el[3] - θ_el[1] # = -detJa_PL*dxi_dr
    Th_21 = θ_el[2] - θ_el[1] # =  detJa_PL*deta_dr

    # the Jacobian ∂ξ∂θ to transform local (ξ, η) into global (θ,r) derivatives
    #     ∂ξ∂θ = [ R_31    -R_21
    #             -Th_31    Th_21] / detJa_PL
    # ∂ξ∂θ = @SMatrix [(r_el[3]-r_el[1]) (r_el[2]-r_el[1])
    #                  (θ_el[3]-θ_el[1])  (θ_el[2]-θ_el[1])]
    Udofs = DoF_U[iel]
    
    # INTEGRATION LOOP
    @inbounds for ip in 1:(A.nip)

        # Unpack shape functions 
        N_ip = N[ip]
        NP_ip = NP[ip]
        N3_ip = N3[ip]
        ∇N_ip = ∇N[ip]

        # ρ at ith integration point
        # ρ_ip = mydot(ρ_el, N3_ip)
        η_ip = _element_viscosity(η, gr.e2n, PhaseID, iel, N3_ip)

        # Polar coordinates of the integration points
        θ_ip = mydot(θ_el, N3_ip)
        r_ip = mydot(r_el, N3_ip)
        cos_ip = cos(θ_ip)
        sin_ip = sin(θ_ip)
        cos_ip_r_ip = cos_ip / r_ip
        sin_ip_r_ip = sin_ip / r_ip

        # Build inverse of the 2nd Jacobian
        # invJ_double = @SMatrix [(∂ξ∂θ[1,1]*cos_ip_r_ip-∂ξ∂θ[2,1]*sin_ip) (-∂ξ∂θ[1,2]*cos_ip_r_ip+∂ξ∂θ[2,2]*sin_ip); 
        #                        -(∂ξ∂θ[1,1]*sin_ip_r_ip+∂ξ∂θ[2,1]*cos_ip) (∂ξ∂θ[1,2]*sin_ip_r_ip+∂ξ∂θ[2,2]*cos_ip) ]
        invJ_double = @SMatrix [
             R_31*cos_ip_r_ip-Th_31*sin_ip   -R_21*cos_ip_r_ip+Th_21*sin_ip
            -R_31*sin_ip_r_ip-Th_31*cos_ip    R_21*sin_ip_r_ip+Th_21*cos_ip
        ]

        # Partial derivatives
        ∂N∂x = invJ_double * ∇N_ip / detJ_pl
        # Integration weight
        ω = r_ip * detJ_pl * w_ip[ip]

        # Update elemental matrices
        D = DMatrix(𝓒, iel, ip, Val(η))
        B = @SMatrix [
            ∂N∂x[1,1]   0           ∂N∂x[1,2]   0           ∂N∂x[1,3]   0           ∂N∂x[1,4]   0           ∂N∂x[1,5]   0           ∂N∂x[1,6]   0
            0           ∂N∂x[2,1]   0           ∂N∂x[2,2]   0           ∂N∂x[2,3]   0           ∂N∂x[2,4]   0           ∂N∂x[2,5]   0           ∂N∂x[2,6]
            ∂N∂x[2,1]   ∂N∂x[1,1]   ∂N∂x[2,2]   ∂N∂x[1,2]   ∂N∂x[2,3]   ∂N∂x[1,3]   ∂N∂x[2,4]   ∂N∂x[1,4]   ∂N∂x[2,5]   ∂N∂x[1,5]   ∂N∂x[2,6]   ∂N∂x[1,6]
        ]

        # Element matrices
        Ke += (B' * D * B) * ω
        Ge += (vec(∂N∂x).*NP_ip) * ω
        Me += (NP_ip.*NP_ip') * (ω/η_ip)

        # Force vector -- right hand side
        dummy1 = @SVector [sin_ip, cos_ip, sin_ip, cos_ip, sin_ip, cos_ip, sin_ip, cos_ip, sin_ip, cos_ip, sin_ip, cos_ip]
        dummy2 = @SVector [N_ip[1], N_ip[1], N_ip[2], N_ip[2], N_ip[3], N_ip[3], N_ip[4], N_ip[4], N_ip[5], N_ip[5], N_ip[6], N_ip[6]]
        fe +=  (g * ω * ρ_ip * cos_ip) * (dummy2 .* dummy1)
        
    end

    # Update global matrices
    update_KS!(KS, Udofs, Ke)
    update_KS!(MS, Pdofs, Me)
    update_GS!(GS, Udofs, Pdofs, Ge)
    update_force_vector!(FS, fe, Udofs)

end