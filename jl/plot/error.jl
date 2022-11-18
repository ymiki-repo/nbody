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


include("../hdf5/nbody.jl")

using PyPlot
include("../util/pyplot.jl")


using Glob
function main()
    # read options
    argv = parse_cmd()
    output_pdf = argv["pdf"]

    # set the target simulation results
    series = argv["target"]
    files = glob(string("dat/", series, "_snp*.h5"))

    # read snapshots
    dat = hdf5_nbody.read_conservatives(files)

    # evaluate Virial ratio
    vir = dat.Ekin ./ -dat.Epot

    # initialize matplotlib
    util_pyplot.config()

    # show time evolution of the energy
    fig_ene = util_pyplot.set_Panel()
    fig_ene.ax[begin].plot(dat.time, dat.Epot, linestyle=util_pyplot.call(fig_ene.line, id=2), color=util_pyplot.call(fig_ene.color, id=2), linewidth=fig_ene.lw, label=L"E_\mathrm{pot}")
    fig_ene.ax[begin].plot(dat.time, dat.Ekin, linestyle=util_pyplot.call(fig_ene.line, id=1), color=util_pyplot.call(fig_ene.color, id=1), linewidth=fig_ene.lw, label=L"E_\mathrm{kin}")
    fig_ene.ax[begin].plot(dat.time, dat.Etot, linestyle=util_pyplot.call(fig_ene.line, id=0), color=util_pyplot.call(fig_ene.color, id=0), linewidth=fig_ene.lw, label=L"E_\mathrm{tot}")
    fig_ene.ax[begin].set_xlabel(string(L"$t$"), fontsize=fig_ene.fs)
    fig_ene.ax[begin].set_ylabel(string(L"$E(t)$"), fontsize=fig_ene.fs)
    fig_ene.ax[begin].grid()
    handles, labels = fig_ene.ax[begin].get_legend_handles_labels()
    fig_ene.ax[begin].legend(handles[end:-1:begin], labels[end:-1:begin], numpoints=1, handlelength=2.0, loc="best", fontsize=fig_ene.fs)

    # show time evolution of the Virial ratio
    fig_vir = util_pyplot.set_Panel()
    fig_vir.ax[begin].plot(dat.time, vir, linestyle=util_pyplot.call(fig_vir.line), color=util_pyplot.call(fig_vir.color), linewidth=fig_vir.lw)
    fig_vir.ax[begin].set_xlabel(string(L"$t$"), fontsize=fig_vir.fs)
    fig_vir.ax[begin].set_ylabel(L"$-K/W$", fontsize=fig_vir.fs)
    fig_vir.ax[begin].grid()

    # check energy conservation
    fig_csv = util_pyplot.set_Panel()
    fig_csv.ax[begin].plot(dat.time, abs.(dat.err_Etot), linestyle=util_pyplot.call(fig_csv.line), color=util_pyplot.call(fig_csv.color), linewidth=fig_csv.lw)
    fig_csv.ax[begin].set_xlabel(string(L"$t$"), fontsize=fig_csv.fs)
    fig_csv.ax[begin].set_ylabel(L"$\abs{E(t) / E(t = 0) - 1}$", fontsize=fig_csv.fs)
    fig_csv.ax[begin].semilogy()
    fig_csv.ax[begin].grid()

    # check linear-momentum conservation
    fig_mom = util_pyplot.set_Panel()
    fig_mom.ax[begin].plot(dat.time, dat.err_px, linestyle=util_pyplot.call(fig_mom.line, id=0), color=util_pyplot.call(fig_mom.color, id=0), linewidth=fig_mom.lw, label=L"$p_x$")
    fig_mom.ax[begin].plot(dat.time, dat.err_py, linestyle=util_pyplot.call(fig_mom.line, id=1), color=util_pyplot.call(fig_mom.color, id=1), linewidth=fig_mom.lw, label=L"$p_y$")
    fig_mom.ax[begin].plot(dat.time, dat.err_pz, linestyle=util_pyplot.call(fig_mom.line, id=2), color=util_pyplot.call(fig_mom.color, id=2), linewidth=fig_mom.lw, label=L"$p_z$")
    fig_mom.ax[begin].set_xlabel(string(L"$t$"), fontsize=fig_mom.fs)
    fig_mom.ax[begin].set_ylabel(string(L"$p(t) - p(t = 0)$"), fontsize=fig_mom.fs)
    fig_mom.ax[begin].yaxis.set_major_formatter(PyPlot.matplotlib.ticker.FuncFormatter(util_pyplot.scientific))
    fig_mom.ax[begin].grid()
    handles, labels = fig_mom.ax[begin].get_legend_handles_labels()
    fig_mom.ax[begin].legend(handles, labels, numpoints=1, handlelength=2.0, loc="best", fontsize=fig_mom.fs)

    # check angular-momentum conservation
    fig_spn = util_pyplot.set_Panel()
    fig_spn.ax[begin].plot(dat.time, dat.err_Lx, linestyle=util_pyplot.call(fig_spn.line, id=0), color=util_pyplot.call(fig_spn.color, id=0), linewidth=fig_spn.lw, label=L"$L_x$")
    fig_spn.ax[begin].plot(dat.time, dat.err_Ly, linestyle=util_pyplot.call(fig_spn.line, id=1), color=util_pyplot.call(fig_spn.color, id=1), linewidth=fig_spn.lw, label=L"$L_y$")
    fig_spn.ax[begin].plot(dat.time, dat.err_Lz, linestyle=util_pyplot.call(fig_spn.line, id=2), color=util_pyplot.call(fig_spn.color, id=2), linewidth=fig_spn.lw, label=L"$L_z$")
    fig_spn.ax[begin].set_xlabel(string(L"$t$"), fontsize=fig_spn.fs)
    fig_spn.ax[begin].set_ylabel(string(L"$L(t) - L(t = 0)$"), fontsize=fig_spn.fs)
    fig_spn.ax[begin].yaxis.set_major_formatter(PyPlot.matplotlib.ticker.FuncFormatter(util_pyplot.scientific))
    fig_spn.ax[begin].grid()
    handles, labels = fig_spn.ax[begin].get_legend_handles_labels()
    fig_spn.ax[begin].legend(handles, labels, numpoints=1, handlelength=2.0, loc="best", fontsize=fig_spn.fs)

    # save figures
    fig_ene.fig.savefig(string("fig/", series, "_energy", ".png"), format="png", dpi=100, bbox_inches="tight")
    fig_vir.fig.savefig(string("fig/", series, "_virial", ".png"), format="png", dpi=100, bbox_inches="tight")
    fig_csv.fig.savefig(string("fig/", series, "_csv_ene", ".png"), format="png", dpi=100, bbox_inches="tight")
    fig_mom.fig.savefig(string("fig/", series, "_csv_mom", ".png"), format="png", dpi=100, bbox_inches="tight")
    fig_spn.fig.savefig(string("fig/", series, "_csv_spn", ".png"), format="png", dpi=100, bbox_inches="tight")
    if output_pdf
        fig_ene.fig.savefig(string("fig/", series, "_energy", ".pdf"), format="pdf", bbox_inches="tight")
        fig_vir.fig.savefig(string("fig/", series, "_virial", ".pdf"), format="pdf", bbox_inches="tight")
        fig_csv.fig.savefig(string("fig/", series, "_csv_ene", ".pdf"), format="pdf", bbox_inches="tight")
        fig_mom.fig.savefig(string("fig/", series, "_csv_mom", ".pdf"), format="pdf", bbox_inches="tight")
        fig_spn.fig.savefig(string("fig/", series, "_csv_spn", ".pdf"), format="pdf", bbox_inches="tight")
    end

    PyPlot.close("all")
    return nothing
end


# using BenchmarkTools
# time = @benchmark main()
# dump(time)

main()
