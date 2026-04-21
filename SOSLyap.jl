using Plots, LaTeXStrings
gr()   # GR backend — high-quality raster output

# This function verifies that a given polynomial V is a valid Lyapunov function for the system defined by f (this only checks positive-semidefiniteness, not strict definiteness)
# Strict definiteness can be enforced by adding a small even monomial term to V and -dV/dt, as shown in the SolveSOSLyapunov function above
function verifyLyapunov(V, x, f)
    model = SOSModel(SCS.Optimizer)
    set_silent(model)
    dV = sum(differentiate(V, x[i]) * f[i] for i in 1:length(x))
    # Check V is SOS
    @constraint(model, V >= 0)
     # Check -dV/dt is SOS
    @constraint(model, -dV >= 0)
    optimize!(model)
    if termination_status(model) == MOI.OPTIMAL
        println("Verification successful: V is a valid Lyapunov function.")
    else
        println("Verification failed: V does not satisfy Lyapunov conditions. Status: ", termination_status(model))
    end
end

function poly_to_scalar(p)
    p isa Number  && return Float64(p)
    iszero(p)     && return 0.0
    return Float64(first(coefficients(p)))
end

## Figures 
function planeLyapunovLevelsAndVectorField(V_simple, f₁, f₂)
    # ── Grid ────────────────────────────────────────────────────────────────────
    Ngrid = 61
    x1_range = range(-2.0, 2.0, length = Ngrid)
    x2_range = range(-2.0, 2.0, length = Ngrid)

    xs = [x1v for x2v in x2_range, x1v in x1_range]
    ys = [x2v for x2v in x2_range, x1v in x1_range]

    # ── Vector field ─────────────────────────────────────────────────────────────
    u_raw = f₁.(xs, ys)
    v_raw = f₂.(xs, ys)

    # Normalize to uniform arrow length for visual clarity
    mag      = sqrt.(u_raw.^2 .+ v_raw.^2) .+ 1e-10
    arrow_len = 0.10
    u_norm   = arrow_len .* u_raw ./ mag
    v_norm   = arrow_len .* v_raw ./ mag


    # ── Lyapunov values on grid ──────────────────────────────────────────────────
    V_grid = [
        poly_to_scalar(subs(V_simple, x[1] => x1v, x[2] => x2v))
        for x2v in x2_range, x1v in x1_range
    ]

    # ── Figure ───────────────────────────────────────────────────────────────────
    default(fontfamily = "Computer Modern")

    # lev = [2, 8, 20, 40, 65, 95, 130]   # hand-picked level-set values
    lev = 8

    p = contour(
        x1_range, x2_range, V_grid;
        levels       = lev,
        linewidth    = 1.6,
        colorbar     = false,
        color        = cgrad(:viridis, lev, categorical=true),
        xlabel       = L"x_1",
        ylabel       = L"x_2",
        xlims        = (-2, 2),
        ylims        = (-2, 2),
        aspect_ratio = :equal,
        framestyle   = :box,
        grid         = false,
        minorticks   = false,
        tick_direction = :out,
        tickfontsize   = 11,
        labelfontsize  = 14,
        size           = (480, 480),
        dpi            = 300,
        legend         = false,
        left_margin    = 4Plots.mm,
        bottom_margin  = 4Plots.mm,
    )

    # ── Quiver on coarser subgrid ─────────────────────────────────────────────────
    stride = 4
    idx_y  = 1:stride:Ngrid
    idx_x  = 1:stride:Ngrid

    quiver!(
        p,
        vec(xs[idx_y, idx_x]),
        vec(ys[idx_y, idx_x]);
        quiver       = (vec(u_norm[idx_y, idx_x]), vec(v_norm[idx_y, idx_x])),
        color        = :black,
        alpha        = 0.6,
        linewidth    = 0.9,
    )

    # Mark the equilibrium
    scatter!(p, [0.0], [0.0];
        markershape      = :circle,
        markersize       = 5,
        markercolor      = :black,
        markerstrokewidth = 0,
    )

    return p
end