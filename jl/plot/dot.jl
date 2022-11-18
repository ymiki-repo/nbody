using ArgParse
function parse_cmd()
    cfg = ArgParseSettings()
    @add_arg_table cfg begin
        "--target", "-t"
        help = "filename of the target files"
        arg_type = String
        default = "collapse"
        "--pdf", "-p"
        help = "generate figure in PDF format"
        action = :store_true
    end
    return parse_args(cfg)
end

using DataFrames
using CSV
function read_csv(file)
    csv = DataFrame(CSV.File(file))
    num = csv[end, :N]
    return num
end

include("../hdf5/nbody.jl")

using PyPlot
include("../util/pyplot.jl")


function draw_Cartesian_map(
    num::Integer, body::hdf5_nbody.Particles, series::String, id::SubString{String};
    output_pdf::Bool=false
)
    fig_pp = util_pyplot.set_Panel(nx=2, ny=2)
    fig_pv = util_pyplot.set_Panel(nx=3, ny=1)

    pt = (num >= 8192) ? "," : "+"
    fig_pp.ax[begin, 2].plot(body.pos[1, begin:end], body.pos[3, begin:end], pt, color="black", markersize=fig_pp.ms)
    fig_pp.ax[begin, 1].plot(body.pos[1, begin:end], body.pos[2, begin:end], pt, color="black", markersize=fig_pp.ms)
    fig_pp.ax[end, 1].plot(body.pos[3, begin:end], body.pos[2, begin:end], pt, color="black", markersize=fig_pp.ms)
    fig_pp.ax[begin, begin].set_xlabel(string(L"$x$"), fontsize=fig_pp.fs)
    fig_pp.ax[end, begin].set_xlabel(string(L"$z$"), fontsize=fig_pp.fs)
    fig_pp.ax[begin, 2].set_ylabel(string(L"$z$"), fontsize=fig_pp.fs)
    fig_pp.ax[begin, 1].set_ylabel(string(L"$y$"), fontsize=fig_pp.fs)
    fig_pp.ax[end, end].set_visible(false)

    fig_pv.ax[1, begin].plot(body.pos[1, begin:end], body.vel[1, begin:end], pt, color="black", markersize=fig_pv.ms)
    fig_pv.ax[2, begin].plot(body.pos[2, begin:end], body.vel[2, begin:end], pt, color="black", markersize=fig_pv.ms)
    fig_pv.ax[3, begin].plot(body.pos[3, begin:end], body.vel[3, begin:end], pt, color="black", markersize=fig_pv.ms)
    fig_pv.ax[begin, begin].set_ylabel(string(L"$v$"), fontsize=fig_pv.fs)
    fig_pv.ax[1, begin].set_xlabel(string(L"$x$"), fontsize=fig_pv.fs)
    fig_pv.ax[2, begin].set_xlabel(string(L"$y$"), fontsize=fig_pv.fs)
    fig_pv.ax[3, begin].set_xlabel(string(L"$z$"), fontsize=fig_pv.fs)

    # set plot domain
    xmin = minimum(body.pos[1, begin:end])
    xmax = maximum(body.pos[1, begin:end])
    ymin = minimum(body.pos[2, begin:end])
    ymax = maximum(body.pos[2, begin:end])
    zmin = minimum(body.pos[3, begin:end])
    zmax = maximum(body.pos[3, begin:end])
    fig_pp.ax[begin, 2].set_ylim(zmin, zmax)
    for ii in 1:2
        fig_pp.ax[ii, 1].set_ylim(ymin, ymax)
    end
    for jj in 1:2
        fig_pp.ax[begin, jj].set_xlim(xmin, xmax)
    end
    vmin = minimum(body.vel)
    vmax = maximum(body.vel)
    for ii in 1:3
        fig_pv.ax[ii, begin].set_ylim(vmin, vmax)
    end
    fig_pv.ax[1, begin].set_xlim(xmin, xmax)
    fig_pv.ax[2, begin].set_xlim(ymin, ymax)
    fig_pv.ax[3, begin].set_xlim(zmin, zmax)

    # add caption
    head = 0
    for jj in fig_pp.ny:-1:1
        for ii in 1:fig_pp.nx
            maptag::String = ""
            if jj == 2 && ii == 1
                maptag = L"$x z$-map"
            elseif jj == 1 && ii == 1
                maptag = L"$x y$-map"
            elseif jj == 1 && ii == 2
                maptag = L"$z y$-map"
            end
            caption = string("(", Char(97 + (ii - 1) + head), ")")
            at = fig_pp.ax[ii, jj]
            at.text(0.05, 0.95, string(caption, "~", maptag), color="black", fontsize=fig_pp.fs, horizontalalignment="left", verticalalignment="top", transform=at.transAxes, bbox=Dict("facecolor" => "white", "edgecolor" => "None", "alpha" => 0.75))
        end
        head += (fig_pp.nx - jj + 1)
    end
    for jj in 1:fig_pv.ny
        for ii in 1:fig_pv.nx
            maptag::String = ""
            if ii == 1
                maptag = L"$x v_x$-map"
            elseif ii == 2
                maptag = L"$y v_y$-map"
            elseif ii == 3
                maptag = L"$z v_z$-map"
            end
            caption = string("(", Char(97 + (ii - 1) + fig_pv.nx * (fig_pv.ny - jj)), ")")
            at = fig_pv.ax[ii, jj]
            at.text(0.05, 0.95, string(caption, "~", maptag), color="black", fontsize=fig_pp.fs, horizontalalignment="left", verticalalignment="top", transform=at.transAxes, bbox=Dict("facecolor" => "white", "edgecolor" => "None", "alpha" => 0.75))
        end
    end

    # save figures
    fig_pp.fig.savefig(string("fig/", series, "_pp", id, ".png"), format="png", dpi=100, bbox_inches="tight")
    fig_pv.fig.savefig(string("fig/", series, "_pv", id, ".png"), format="png", dpi=100, bbox_inches="tight")
    if output_pdf
        fig_pp.fig.savefig(string("fig/", series, "_pp", id, ".pdf"), format="pdf", bbox_inches="tight")
        fig_pv.fig.savefig(string("fig/", series, "_pv", id, ".pdf"), format="pdf", bbox_inches="tight")
    end

    fig_pp = nothing
    fig_pv = nothing
    PyPlot.close("all")

    return nothing
end


using FilePaths
using Glob
using MPI
function main()
    MPI.Init()
    size = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    # read options
    argv = parse_cmd()
    output_pdf = argv["pdf"]

    # set the series of simulation results
    series = argv["target"]
    files = glob(string("dat/", series, "_snp*.h5"))
    Nfiles = length(files)
    num = read_csv(string("log/", series, "_run.csv"))

    # initialize matplotlib
    util_pyplot.config()

    for fid in (1+rank):size:Nfiles
        # read snapshot
        file = files[fid]
        dat = hdf5_nbody.read_particle(file, num)
        name, sid = split(filename(Path(file)), "_snp")

        draw_Cartesian_map(num, dat, series, sid, output_pdf=output_pdf)
        dat = nothing
    end

    MPI.Finalize()
    return nothing
end


main()
