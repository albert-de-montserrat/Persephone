
function _stress!(
    F,
    U,
    iel,
    DoF_U,
    θ,
    r,
    nip,
    SF_Stokes::ShapeFunctionsStokes,
)
    # g, 
    # ρ, 
    # η, 
    # 𝓒,

    N, ∇N, NP, dN3ds, w_ip, N3 = 
        SF_Stokes.N, SF_Stokes.∇N, SF_Stokes.NP, SF_Stokes.dN3ds, SF_Stokes.w_ip, SF_Stokes.N3 
    
    # Polar coordinates of element nodes
    θ_el = @SVector [θ[i, iel] for i in 1:3]
    r_el = @SVector [r[i, iel] for i in 1:3]
    coords = SMatrix{3,2}([θ_el r_el])

    # Jacobian n. 1 (p:=polar, l:=local): reference element --> current element
    J_pl = dN3ds * coords
    detJ_pl = mydet(J_pl)

    # the Jacobian ∂ξ∂θ to transform local (ξ, η) into global (θ,r) derivatives
    #     ∂ξ∂θ = [ R_31    -R_21
    #             -Th_31    Th_21] / detJa_PL
    R_21 = r_el[2] - r_el[1]  # = -detJa_PL*deta_dth
    R_31 = r_el[3] - r_el[1]  # =  detJa_PL*dxi_dth
    Th_31 = θ_el[3] - θ_el[1] # = -detJa_PL*dxi_dr
    Th_21 = θ_el[2] - θ_el[1] # =  detJa_PL*deta_dr

    Udofs = DoF_U[iel]
    U_el = @SVector [
        U[Udofs[1]], U[Udofs[2]], U[Udofs[3]], U[Udofs[4]],  U[Udofs[5]],  U[Udofs[6]],
        U[Udofs[7]], U[Udofs[8]], U[Udofs[9]], U[Udofs[10]], U[Udofs[11]], U[Udofs[12]]
    ]

    # INTEGRATION LOOP
    @inbounds for ip in 1:nip

        # Unpack shape functions 
        # N_ip = N[ip]
        # NP_ip = NP[ip]
        N3_ip = N3[ip]
        ∇N_ip = ∇N[ip]

        # ρ at ith integration point
        # ρ_ip = mydot(ρ_el, N3_ip)
        # η_ip = _element_viscosity(η, gr.e2n, PhaseID, iel, N3_ip)

        # Polar coordinates of the integration points
        θ_ip = mydot(θ_el, N3_ip)
        r_ip = mydot(r_el, N3_ip)
        sin_ip, cos_ip = sincos(θ_ip)
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

        # Update elemental matrices
        # D = DMatrix(𝓒, iel, ip, Val(η)) 
        # Use this to compute directly strain rate (i.e. B*Uel)
        # B = @SMatrix [
        #     ∂N∂x[1,1]   0           ∂N∂x[1,2]   0           ∂N∂x[1,3]   0           ∂N∂x[1,4]   0           ∂N∂x[1,5]   0           ∂N∂x[1,6]   0
        #     0           ∂N∂x[2,1]   0           ∂N∂x[2,2]   0           ∂N∂x[2,3]   0           ∂N∂x[2,4]   0           ∂N∂x[2,5]   0           ∂N∂x[2,6]
        #     ∂N∂x[2,1]   ∂N∂x[1,1]   ∂N∂x[2,2]   ∂N∂x[1,2]   ∂N∂x[2,3]   ∂N∂x[1,3]   ∂N∂x[2,4]   ∂N∂x[1,4]   ∂N∂x[2,5]   ∂N∂x[1,5]   ∂N∂x[2,6]   ∂N∂x[1,6]
        # ]

        # modified B to calculcate partial derivatives of velocity
        B = @SMatrix [
            ∂N∂x[1,1]   0.0        ∂N∂x[1,2]   0.0        ∂N∂x[1,3]    0.0         ∂N∂x[1,4]   0.0         ∂N∂x[1,5]   0.0         ∂N∂x[1,6]   0.0
            0.0         ∂N∂x[2,1]   0.0         ∂N∂x[2,2]   0.0         ∂N∂x[2,3]   0.0         ∂N∂x[2,4]   0.0         ∂N∂x[2,5]   0.0         ∂N∂x[2,6]
            ∂N∂x[2,1]   0.0         ∂N∂x[2,2]   0.0         ∂N∂x[2,3]   0.0         ∂N∂x[2,4]   0.0         ∂N∂x[2,5]   0.0         ∂N∂x[2,6]   0.0        
            0.0         ∂N∂x[1,1]   0.0         ∂N∂x[1,2]   0.0         ∂N∂x[1,3]   0.0         ∂N∂x[1,4]   0.0         ∂N∂x[1,5]   0.0         ∂N∂x[1,6]
        ]

        ∂U∂x = B*U_el # [∂Ux∂x ∂Uz∂z ∂Ux∂z ∂Uz∂x]
        # transpose of the velocity gradient
        ∇Uᵀ = @SMatrix [
            ∂U∂x[1] ∂U∂x[3]
            ∂U∂x[4] ∂U∂x[2]
        ]

        F0 = F[iel, ip]
        k1 = _F_Rk4(∇Uᵀ, F0)
        Fi = k1*0.5 .+ F0
        k2 = _F_Rk4(∇Uᵀ, Fi)
        Fi = k2*0.5 .+ F0
        k3 = _F_Rk4(∇Uᵀ, Fi)
        Fi = k3 .+ F0
        k4 = _F_Rk4(∇Uᵀ, Fi)
        F[iel, ip] = F0 + (k1 + 2*(k2 + k3) +k4)/6
        
    end

end

_F_Rk4(∇Uᵀ, F) = ∇Uᵀ*F

function stress!( 
    F,
    U,
    nel,
    DoF_U,
    θ,
    r,
    nip,
    SF_Stokes
)
    @batch per=core for iel in 1:nel
        # @code_warntype
        _stress!(
            F,
            U,
            iel,
            DoF_U,
            θ,
            r,
            nip,
            SF_Stokes,
        )
    end
end