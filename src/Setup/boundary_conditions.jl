struct BC{T,N}
    Ω::Vector{T}
    ifree::Vector{T}
    vfix::Vector{N}
end

function init_BCs(gr, IDs; Ttop = 0.0, Tbot = 1.0, type="free slip" )
    TBC = temperature_bcs(gr, IDs; Ttop = Ttop, Tbot =Tbot)
    UBC = velocity_bcs(gr, IDs; type=type)
    return TBC, UBC
end

function velocity_bcs(M::Grid, nr, ids; type="free slip")
    if type == "free slip"
        Ωu, ufix = free_slip(nr, ids)
    elseif type == "no slip"
        Ωu, ufix = no_slip(ids)
    end

    nU = maximum(M.e2n)
    ifree = setdiff(collect(1:2nU),Ωu)
    BC(Ωu, ifree, ufix)
end

function velocity_bcs(M::Grid, ids; type="free slip")
    if type == "free slip"
        Ωu, ufix = free_slip(ids)
    elseif type == "no slip"
        Ωu, ufix = no_slip(ids)
    end

    nU = maximum(M.e2n)
    ifree = setdiff(collect(1:2nU),Ωu)
    BC(Ωu, ifree, ufix)
end

function velocity_bcs(nU::Int, ids; type="free slip")
    if type == "free slip"
        Ωu, ufix = free_slip(ids)
    elseif type == "no slip"
        Ωu, ufix = no_slip(ids)
    end

    # nU = maximum(e2n)
    ifree = setdiff(collect(1:2nU),Ωu)
    BC(Ωu, ifree, ufix)
end

function free_slip(ids)
    top_nodes = findall(x->x == "outter", ids)
    bot_nodes = findall(x->x == "inner", ids)
    nodes = vcat(top_nodes, bot_nodes)
    
    # fix radial velocity only
    Ωu = @. 2*nodes
    ufix = fill(0.0, length(Ωu))

    return Ωu, ufix
end

function free_slip(nr::Int, ids)
    top_nodes = findall(x->x == "outter", ids)
    bot_nodes = findall(x->x == "inner", ids)
    nodes = vcat(top_nodes, bot_nodes)
    
    # fix radial velocity only
    Ωu = @. 2*nodes
    ufix = fill(0.0, length(Ωu))

    # fix tangential velocity of one single node
    push!(Ωu,2*nr-1)
    push!(ufix, 0.0)

    return Ωu, ufix
end

function no_slip(ids)
    nodes = findall(x-> (x == "outter") || (x == "inner"), ids)
    # bot_nodes = findall(x-> x == "inner", ids)
    # nodes = vcat(top_nodes, bot_nodes)
    
    # fix radial velocity only
    Ωu1 = @. 2*nodes
    Ωu2 = @. 2*nodes-1
    Ωu = vcat(Ωu1, Ωu2)
    ufix = fill(0.0, length(Ωu))

    return Ωu, ufix
end

function temperature_bcs(M::Grid, ids; Ttop=0.0, Tbot=1.0)
    # Top nodes & temperature
    top_nodes = findall(x->x == "outter", ids)
    top_T = fill(Ttop, length(top_nodes))

    # Top nodes & temperature
    bot_nodes = findall(x->x == "inner", ids)
    bot_T = fill(Tbot, length(top_nodes))

    # Concatenate DoFs and T
    ΩT = vcat(top_nodes, bot_nodes)
    tfix = vcat(top_T, bot_T)

    nT = length(M.x)
    tfree = setdiff(collect(1:nT),ΩT)
    BC(ΩT, tfree, tfix)
end