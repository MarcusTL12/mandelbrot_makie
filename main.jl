using Images
using SIMD

using GLMakie
include("ui.jl")

const Vec = SIMD.Vec

function counts2HSV(counts, max_n)
    img = zeros(HSV{Float64}, size(counts))
    img_mat = reshape(reinterpret(Float64, img), 3, size(counts)...)
    H = @view img_mat[1, :, :]
    S = @view img_mat[2, :, :]
    V = @view img_mat[3, :, :]

    V .= 1.0
    S .= 1.0

    H .= counts
    H .*= 60 / 256
    H .+= 240

    for i in eachindex(counts)
        if counts[i] == max_n
            V[i] = 0.0
        end
    end

    img
end

# Reference Implementation
function mandel_many(c, max_n)
    cnt = 0

    z = 0.0im

    @fastmath for _ in 1:max_n
        if abs2(z) > 4.0
            break
        end

        z = z^2 + c
        cnt += 1
    end

    z, cnt
end

function mandel_ref!(c_min, c_max, counts, max_n)
    cr_range = range(real(c_min), real(c_max), length=size(counts, 2))
    ci_range = range(imag(c_min), imag(c_max), length=size(counts, 1))

    @inbounds @fastmath for (j, cr) in enumerate(cr_range)
        for (i, ci) in enumerate(cr_range)
            counts[i, j] = @inline mandel_many(cr + ci * im, max_n)[2]
        end
    end
end

function mandel_simd_many(cr, ci, zr, zi, max_n)
    counts = Vec((0 for _ in Tuple(cr))...,)

    v_ones = Vec((1 for _ in Tuple(cr))...,)
    v_zeros = Vec((0 for _ in Tuple(cr))...,)

    @fastmath for _ in 1:max_n
        zr2 = zr^2
        zi2 = zi^2

        mask = zr2 + zi2 < 4.0
        if !any(mask)
            break
        end

        counts += vifelse(mask, v_ones, v_zeros)
        zi = vifelse(mask, 2 * zr * zi + ci, zi)
        zr = vifelse(mask, zr2 - zi2 + cr, zr)
    end

    zr, zi, counts
end

@noinline function mandel!(c_min, c_max, zr, zi, counts, max_n,
    ::Val{N}) where {N}
    @assert size(zr) == size(zi)
    @assert size(zr) == size(counts)
    @assert size(zr, 1) % N == 0

    cr_range = collect(range(real(c_min), real(c_max), length=size(zr, 2)))
    ci_range = collect(range(imag(c_min), imag(c_max), length=size(zr, 1)))

    for (j, cr) in enumerate(cr_range)
        cr_v = Vec{N,typeof(cr)}(cr)
        @inbounds @fastmath for i in 1:N:length(ci_range)
            i_v = VecRange{N}(i)
            ci_v = ci_range[i_v]
            zr_v = zr[i_v, j]
            zi_v = zi[i_v, j]
            counts_v = counts[i_v, j]

            zr_v, zi_v, counts_v =
                @inline mandel_simd_many(cr_v, ci_v, zr_v, zi_v, max_n)

            zr[i_v, j] = zr_v
            zi[i_v, j] = zi_v
            counts[i_v, j] += counts_v
        end
    end
end

function mandel_thread!(c_min, c_max, zr, zi, counts, max_n, ::Val{N}) where {N}
    cr_range = range(real(c_min), real(c_max), length=size(zr, 2))

    nth = Threads.nthreads()
    @assert length(cr_range) % nth == 0

    blocksize = length(cr_range) ÷ nth

    Threads.@threads for id in 1:nth
        index_range = ((id-1)*blocksize+1):(id*blocksize)

        cr_subrange = cr_range[index_range]
        zr_v = @view zr[:, index_range]
        zi_v = @view zi[:, index_range]
        counts_v = @view counts[:, index_range]

        cr_min = cr_subrange[begin]
        cr_max = cr_subrange[end]

        c_min_local = cr_min + imag(c_min) * im
        c_max_local = cr_max + imag(c_max) * im

        mandel!(c_min_local, c_max_local, zr_v, zi_v, counts_v, max_n, Val(N))
    end
end

function make_mandel_image(lims, resolution, max_n, ::Val{N}) where {N}
    res = reverse(resolution)

    zr = zeros(res)
    zi = zeros(res)
    counts = zeros(Int, res)

    c_min = lims[1][1] + lims[2][1] * im
    c_max = lims[1][2] + lims[2][2] * im

    # @time mandel_ref!(c_min, c_max, counts, max_n)
    # @time mandel!(c_min, c_max, zr, zi, counts, max_n, Val(N))
    @time mandel_thread!(c_min, c_max, zr, zi, counts, max_n, Val(N))

    counts2HSV(counts, max_n)
end
