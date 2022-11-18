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

using Parameters
@with_kw mutable struct Performance
    num::Array{Integer,1}
    time_per_step::Array{Real,1}
    interactions_per_sec::Array{Real,1}
    Flops::Array{Real,1}
    FP_L::Array{Integer,1}
    FP_M::Array{Integer,1}
end

using DataFrames
using CSV
function read_csv(file)
    csv = DataFrame(CSV.File(file))
    results = Performance(
        num=csv[!, :N],
        time_per_step=csv[!, :"time_per_step[s]"],
        interactions_per_sec=csv[!, :interactions_per_sec],
        Flops=csv[!, :"Flop/s"],
        FP_L=csv[!, :"FP_L"],
        FP_M=csv[!, :"FP_M"]
    )
    return results
end


using PyPlot
include("../util/pyplot.jl")


function main()
    # read options
    argv = parse_cmd()
    output_pdf = argv["pdf"]

    # find the latest series of simulation results
    series = argv["target"]

    # read CSV file
    dat = read_csv(string("log/", series, "_run.csv"))

    # initialize matplotlib
    util_pyplot.config()

    # show observed performance as a function of N
    fig_perf = util_pyplot.set_Panel()
    fig_perf.ax[begin].plot(dat.num[(dat.FP_L.==32).&&(dat.FP_M.==32)], 1.0e-12 .* dat.Flops[(dat.FP_L.==32).&&(dat.FP_M.==32)], util_pyplot.call(fig_perf.point, id=0), color=util_pyplot.call(fig_perf.color, id=0), markersize=fig_perf.ms, linestyle=util_pyplot.call(fig_perf.line, id=0), linewidth=fig_perf.lw, label="Low: FP32, Mid: FP32")
    fig_perf.ax[begin].plot(dat.num[(dat.FP_L.==32).&&(dat.FP_M.==64)], 1.0e-12 .* dat.Flops[(dat.FP_L.==32).&&(dat.FP_M.==64)], util_pyplot.call(fig_perf.point, id=1), color=util_pyplot.call(fig_perf.color, id=1), markersize=fig_perf.ms, linestyle=util_pyplot.call(fig_perf.line, id=1), linewidth=fig_perf.lw, label="Low: FP32, Mid: FP64")
    fig_perf.ax[begin].plot(dat.num[(dat.FP_L.==64).&&(dat.FP_M.==64)], 1.0e-12 .* dat.Flops[(dat.FP_L.==64).&&(dat.FP_M.==64)], util_pyplot.call(fig_perf.point, id=2), color=util_pyplot.call(fig_perf.color, id=2), markersize=fig_perf.ms, linestyle=util_pyplot.call(fig_perf.line, id=2), linewidth=fig_perf.lw, label="Low: FP64, Mid: FP64")
    fig_perf.ax[begin].set_xlabel(L"$N$", fontsize=fig_perf.fs)
    fig_perf.ax[begin].set_ylabel(L"$\unit{TFlop/s}$", fontsize=fig_perf.fs)
    # fig_perf.ax[begin].loglog()
    fig_perf.ax[begin].semilogx()
    fig_perf.ax[begin].grid()
    handles, labels = fig_perf.ax[begin].get_legend_handles_labels()
    fig_perf.ax[begin].legend(handles[end:-1:begin], labels[end:-1:begin], numpoints=1, handlelength=2.0, loc="best", fontsize=fig_perf.fs)

    # save figures
    fig_perf.fig.savefig(string("fig/", series, "_scl_perf", ".png"), format="png", dpi=100, bbox_inches="tight")
    if output_pdf
        fig_perf.fig.savefig(string("fig/", series, "_scl_perf", ".pdf"), format="pdf", bbox_inches="tight")
    end

    fig_perf = nothing
    PyPlot.close("all")
    return nothing
end


main()
