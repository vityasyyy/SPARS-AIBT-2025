let
  kapack = import (fetchTarball "https://github.com/oar-team/nur-kapack/archive/master.tar.gz") {};

  # Folder lokal batsched bisa kamu edit sesukanya
  localBatschedSrc = builtins.path {
    path = /papers/HPCv2/batsim/batsched;
    name = "batsched-src";
  };
in

kapack.pkgs.mkShell {
  name = "dev-env";
  buildInputs = with kapack.pkgs; [
    kapack.batsim           # binary simulator
    cmake
    boost
    gmp
    rapidjson
    hiredis
    libev
    zeromq
    cppzmq
    kapack.intervalset      # <- penting ini, previously missing!
    kapack.loguru           # optional, but usually used
    kapack.redox  # 🔥 tambahkan ini
  ];

  shellHook = ''
    echo "✅ Ready to build batsched manually from ${localBatschedSrc}"
    echo "💡 cd ${localBatschedSrc} && mkdir -p build && cd build && cmake .. && make"
  '';
}
