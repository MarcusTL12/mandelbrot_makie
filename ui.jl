
function get_axis_dimensions(ax, ::Val{N}) where {N}
    coord_limits =
        ax.xaxis.attributes.limits[],
        ax.yaxis.attributes.limits[]

    x_endpoints = ax.xaxis.attributes.endpoints[]
    y_endpoints = ax.yaxis.attributes.endpoints[]
    x_resolution = ceil(Int, x_endpoints[2][1] - x_endpoints[1][1] + 1)
    y_resolution = ceil(Int, y_endpoints[2][2] - y_endpoints[1][2] + 1)

    coord_limits, roundoff_resolution((x_resolution, y_resolution), Val(N))
end

function roundoff_resolution((x, y), ::Val{N}) where {N}
    if y % N != 0
        y = (y ÷ N + 1) * N
    end

    nth = Threads.nthreads()

    if x % nth != 0
        x = (x ÷ nth + 1) * nth
    end

    (x, y)
end

function mandelgui(::Val{N}) where {N}
    f = Figure()

    ax = GLMakie.Axis(f[1:2, 1], aspect=DataAspect(),
        limits=((-2.0, 2.0), (-2.0, 2.0)))

    (x_lims, y_lims), resolution = get_axis_dimensions(ax, Val(N))

    x_lims = Observable(x_lims)
    y_lims = Observable(y_lims)

    recompute = Button(f[2, 2][1, 1], label="recompute")
    iter_box = Textbox(f[2, 2][2, 1], placeholder="1000")

    iterations = Observable(1000)

    on(iter_box.stored_string) do s
        iterations[] = parse(Int, s)
    end

    img = Observable(make_mandel_image((x_lims[], y_lims[]), resolution,
        iterations[], Val(N))')

    image!(ax, x_lims, y_lims, img; interpolate=false)

    on(recompute.clicks) do _
        (x_lims[], y_lims[]), resolution = get_axis_dimensions(ax, Val(N))

        @show resolution

        img[] = make_mandel_image((x_lims[], y_lims[]), resolution,
            iterations[], Val(N))'
    end

    f
end

mandelgui() = mandelgui(Val(16))
