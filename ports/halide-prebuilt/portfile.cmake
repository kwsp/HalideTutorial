vcpkg_check_linkage(ONLY_DYNAMIC_LIBRARY)


# Define the URL of the archive
set(VERSION v18.0.0)
set(URL_BASE "https://github.com/halide/Halide/releases/download/${VERSION}")

set(ARCHIVE_NAME NOTFOUND)
set(SHA512 0)
if(VCPKG_TARGET_IS_WINDOWS)
  set(ARCHIVE_NAME "Halide-18.0.0-x86-64-windows-8c651b459a4e3744b413c23a29b5c5d968702bb7.zip")
  set(SHA512 3075fb339f17fdcf7dcb60a61d464f173b67a3ae467efe60bd387f46d0d386de83bfe2b6ddbb8c73a17a4c1538049294613557c2143cfd997a5879319858ef80)

elseif(VCPKG_TARGET_IS_OSX)
  set(ARCHIVE_NAME "Halide-18.0.0-arm-64-osx-8c651b459a4e3744b413c23a29b5c5d968702bb7.tar.gz")
  set(SHA512 d85dd21e0bcc8549d64b562013944f7efb696750f837ae327fadb2928578e1bc5e56896d78817cd726703818af01d1ed1be8317c641436a884a473151f4074a5)

endif()

# Download the archive using vcpkg_download_distfile
vcpkg_download_distfile(
    OUTFILE
    URLS ${URL_BASE}/${ARCHIVE_NAME}
    SHA512 ${SHA512}
    FILENAME ${ARCHIVE_NAME}
)

# Extract the downloaded archive
vcpkg_extract_source_archive_ex(
    OUT_SOURCE_PATH SOURCE_PATH
    ARCHIVE ${OUTFILE}
    REF "Halide-${VERSION}"
)

file(INSTALL ${SOURCE_PATH}/include/ DESTINATION ${CURRENT_PACKAGES_DIR}/include/)
file(INSTALL ${SOURCE_PATH}/lib/ DESTINATION ${CURRENT_PACKAGES_DIR}/lib/)
file(INSTALL ${SOURCE_PATH}/bin/ DESTINATION ${CURRENT_PACKAGES_DIR}/bin/)

vcpkg_copy_pdbs()

file(INSTALL ${SOURCE_PATH}/share/Halide/ DESTINATION ${CURRENT_PACKAGES_DIR}/share/Halide/)

# vcpkg_cmake_config_fixup(PACKAGE_NAME Halide)
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/share/doc/Halide/LICENSE.txt")
