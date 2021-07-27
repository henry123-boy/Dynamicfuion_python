set(U2NET_PRETRAINED_MODEL_DOWNLOADED FALSE)
set(U2NET_PRETRAINED_MODEL_EXPECTED_CHECKSUM 10025a17f49cd3208afc342b589890e402ee63123d6f2d289a4a0903695cce58)
if(EXISTS ${NNRT_3RDPARTY_DIR}/U-2-Net/saved_models/u2net.pth)
    file(SHA256 ${NNRT_3RDPARTY_DIR}/U-2-Net/saved_models/u2net.pth DOWNLOAD_CHECKSUM)
    if(DOWNLOAD_CHECKSUM STREQUAL U2NET_PRETRAINED_MODEL_EXPECTED_CHECKSUM)
        set(U2NET_PRETRAINED_MODEL_DOWNLOADED TRUE)
    endif()
endif()

if(NOT U2NET_PRETRAINED_MODEL_DOWNLOADED)
    file(MAKE_DIRECTORY ${NNRT_3RDPARTY_DIR}/U-2-Net/saved_models)
    file(DOWNLOAD https://algomorph.com/storage/reco/pretrained_models/u2net.pth
        ${NNRT_3RDPARTY_DIR}/U-2-Net/saved_models/u2net.pth SHOW_PROGRESS STATUS DOWNLOAD_STATUS
    )
    message(STATUS "DOWNLOAD_STATUS: ${DOWNLOAD_STATUS}")
endif()