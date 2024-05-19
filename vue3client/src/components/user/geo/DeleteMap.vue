<template>
    <div>
        <div class="modal fade" id="modal-delete-map" tabindex="-1" role="dialog" aria-labelledby="exampleModalLabel"
            aria-hidden="true" @click="closeModal()">
            <div class="modal-dialog" role="document">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title" id="exampleModalLabel"><i class="fa-solid fa-triangle-exclamation"></i>
                            Warning</h5>
                        <button type="button" class="close" data-dismiss="modal" aria-label="Close"
                            @click="resetData()">
                            <span aria-hidden="true"><i class="fa-regular fa-circle-xmark"></i></span>
                        </button>
                    </div>
                    <div class="modal-body">
                        <div class="alert alert-warning" role="alert">
                            <p class="mb-2">Warning: This image will be permanently deleted from the system !</p>
                        </div>
                        <div class="row">
                            <div class="col-6">
                                <div class="form-group">
                                    <label for="exampleFormControlInput1">Address</label>
                                    <input type="text" readonly class="form-control" id="exampleFormControlInput1"
                                        v-model="mapSelected.address">
                                </div>
                            </div>
                            <div class="col-6">
                                <div class="form-group">
                                    <label for="exampleFormControlInput1">Email address</label>
                                    <input type="text" readonly v-model="mapSelected.coordinates" class="form-control"
                                        id="exampleFormControlInput1">
                                </div>
                            </div>
                        </div>
                        <div>
                            <img :src="mapSelected.imagefull ? mapSelected.imagefull : require('@/assets/pandora360.png')"
                                alt="">
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-outline-secondary" data-dismiss="modal" ref="closeButton"
                            id="close" @click="resetData()">Close</button>
                        <button type="button" class="btn btn-outline-danger" @click="deleteBook"><i
                                class="fa-solid fa-trash"></i> Delete </button>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script>

import UserRequest from '@/restful/UserRequest';
import useEventBus from '@/composables/useEventBus';
const { emitEvent, onEvent } = useEventBus();

export default {
    name: "DeleteMap",

    props: {
    },
    data() {
        return {
            mapSelected: {
                id: null,
                address: null,
                coordinates: null,
                imagefull: null,
                pz: null,
                nz: null,
                px: null,
                nx: null,
                py: null,
                ny: null,
            },
            dataSubmit: {
                is_delete: '',
            }
        }
    },
    mounted() {
        onEvent('selectSimpleMapDelete', (mapSelected) => {
            this.mapSelected = mapSelected;
        });
    },
    methods: {
        deleteBook: async function () {
            try {
                if (this.mapSelected.is_delete == 0) this.dataSubmit.is_delete = 1;
                else this.dataSubmit.is_delete = 0;
                const { messages } = await UserRequest.post('map/delete/' + this.mapSelected.id, this.dataSubmit, true);
                emitEvent('eventSuccess', messages[0]);
                const closeButton = this.$refs.closeButton;
                closeButton.click();
                emitEvent('eventRegetMaps', '');
            }
            catch (error) {
                if (error.messages) emitEvent('eventError', error.messages[0]);
            }
        },
        closeModal: function () {
            if (event.target.classList.contains('modal')) {
                this.resetData();
            }
        },
        resetData: function () {
            // this.mapSelected = {
            //     id : null,
            //     address : null,
            //     coordinates : null,
            //     imagefull: null,
            //     pz : null,
            //     nz : null,
            //     px : null,
            //     nx : null,
            //     py : null,
            //     ny : null,
            // };
        },
    }

}
</script>

<style scoped>
.modal-header .close {
    outline: none;
}

.imgInTable img {
    max-width: 100px;
}

.innerData {
    display: flex;
    align-items: center;
    align-content: center;
    margin-top: 10px;
}

.innerData img {
    margin-left: 3px;
    border-radius: 6px;
}

.modal-dialog {
    max-width: 600px;
}

.contentText {
    color: var(--user-color);
}

.colorImage {
    color: var(--blue-color);
}

.colorText {
    color: var(--brown-color)
}

@media screen and (min-width: 993px) and (max-width: 1200px) {
    .modal-dialog {
        max-width: 400px;
        margin: 10px auto;
        font-size: 12px;
    }

    .modal-header {
        padding: auto;
    }

    .modal-header .close {
        font-size: 20px;
    }

    .btn {
        font-size: 13px;
    }
}

@media screen and (min-width: 769px) and (max-width: 992px) {
    .modal-dialog {
        max-width: 350px;
        margin: 10px auto;
        font-size: 11px;
    }

    .modal-header {
        padding: auto;
    }

    .modal-header .close {
        font-size: 18px;
    }

    .btn {
        font-size: 12px;
    }

    .modal-body {
        padding: 14px 14px 0 14px;
    }
}

@media screen and (min-width: 577px) and (max-width: 768px) {
    .modal-dialog {
        max-width: 320px;
        margin: 10px auto;
        font-size: 9px;
    }

    .modal-header {
        padding: auto;
    }

    .modal-header .close {
        font-size: 11px;
    }

    .btn {
        font-size: 10px;
    }

    .modal-body {
        padding: 14px 14px 0 14px;
    }

    .alert {
        padding: 8px;
    }

    .imgInTable img {
        max-width: 90px;
    }
}

@media screen and (min-width: 425px) and (max-width: 575px) {
    .modal-dialog {
        max-width: 275px;
        margin: 10px auto;
        font-size: 9px;
    }

    .modal-header,
    .modal-footer {
        padding: 5px 5px;
    }

    .modal-header .close {
        font-size: 11px;
    }

    .btn {
        font-size: 8px;
    }

    .modal-body {
        padding: 12px 12px 0 12px;
    }

    .alert {
        padding: 8px;
    }

    .imgInTable img {
        max-width: 80px;
    }
}

@media screen and (min-width: 375px) and (max-width: 424px) {
    .modal-dialog {
        max-width: 180px;
        margin: 10px auto;
        font-size: 7px;
    }

    .modal-header,
    .modal-footer {
        padding: 5px 5px;
    }

    .modal-header .close {
        font-size: 9px;
    }

    .btn {
        font-size: 7px;
    }

    .modal-body {
        padding: 11px 11px 0 11px;
    }

    .alert {
        padding: 6px 10px;
    }

    .imgInTable img {
        max-width: 70px;
    }

    .ml-3 {
        margin-left: 6px !important;
    }

    .mb-2 {
        margin-bottom: 0.5px !important
    }
}
</style>