<template>
    <div>
        <div id="big">
            <div class="bigContainer">
                <div class="modal fade" id="modal-add-map" tabindex="-1" role="dialog"
                    aria-labelledby="exampleModalLabel" aria-hidden="true">
                    <div class="modal-dialog" role="document">
                        <div class="modal-content">
                            <div class="modal-header">
                                <h5 class="modal-title" id="exampleModalLabel"><strong><i
                                            class="fa-solid fa-envelope-open-text"></i>
                                        Add Map</strong></h5>
                                <button type="button" class="close" data-dismiss="modal" aria-label="Close">
                                    <span aria-hidden="true" class="text-danger"><i
                                            class="fa-regular fa-circle-xmark"></i></span>
                                </button>
                            </div>
                            <div class="modal-body">
                                <form @submit.prevent="addMap()">
                                    <legend>Import Image Pandora 360</legend>
                                    <div class="form-group row">
                                        <label for="inputPassword" class="col-sm-2 col-form-label">Address</label>
                                        <div class="col-sm-10">
                                            <input required v-model="submitAdress" type="text" class="form-control"
                                                id="inputPassword" placeholder="Address">
                                        </div>
                                    </div>
                                    <div class="form-group row">
                                        <label for="inputPassword" class="col-sm-2 col-form-label">Coordinates</label>
                                        <div class="col-sm-10">
                                            <input required v-model="submitCoordinates" type="text" class="form-control"
                                                id="inputPassword" placeholder="Coordinates">
                                        </div>
                                    </div>
                                    <div class="loadContent">
                                        <div class="minAvatar">
                                            <input id="imageInput" required class="input-file" type="file"
                                                @change="previewImage" accept="image/*" ref="fileInput" />
                                            <span class="iconClound" v-if="previewImageSrc == null"><i
                                                    class="fa-solid fa-cloud-arrow-up"></i></span>
                                            <div v-if="previewImageSrc" class="box-preview">
                                                <img class="preview" :src="previewImageSrc" alt="Preview" />
                                                <img src="@/assets/error.png" @click="removeFile" class="close"
                                                    alt="Remove">
                                            </div>
                                        </div>
                                    </div>
                                </form>
                                <br>
                                <section class="settings">
                                    <div class="form-group row">
                                        <label for="inputPassword" class="col-sm-2 col-form-label">Cube Rotation</label>
                                        <div class="col-sm-2">
                                            <input class="form-control" id="cubeRotation" type="number" min="0"
                                                max="359" v-model="rotation" @change="loadImage" />
                                        </div>
                                    </div>
                                    <fieldset title="The resampling algorithm to use when generating the cubemap.">
                                        <legend>Interpolation type</legend>
                                        <div class="row">
                                            <div class="col-4">
                                                <label>
                                                    <input type="radio" name="interpolation" value="lanczos"
                                                        v-model="interpolation" @change="loadImage" /> Lanczos (best but
                                                    slower)
                                                </label>
                                            </div>
                                            <div class="col-4">
                                                <label>
                                                    <input type="radio" name="interpolation" value="cubic"
                                                        v-model="interpolation" @change="loadImage" /> Cubic (sharper
                                                    details)
                                                </label>
                                            </div>
                                            <div class="col-4">
                                                <label>
                                                    <input type="radio" name="interpolation" value="linear"
                                                        v-model="interpolation" @change="loadImage" /> Linear (softer
                                                    details)
                                                </label>
                                            </div>
                                        </div>
                                    </fieldset>
                                    <fieldset>
                                        <legend>Output format</legend>
                                        <div class="row">
                                            <div class="col-2">
                                                <label>
                                                    <input type="radio" name="format" value="png" v-model="format"
                                                        @change="loadImage" /> PNG
                                                </label>
                                            </div>
                                            <div class="col-2">
                                                <label>
                                                    <input type="radio" name="format" value="jpg" v-model="format"
                                                        @change="loadImage" /> JPEG
                                                </label>
                                            </div>
                                        </div>
                                    </fieldset>
                                </section>
                                <section>
                                    <p v-if="!generating && faces.length > 0">Click each cube face to save it to your computer.</p>
                                    <flower-spinner v-if="generating" class="loading-component"
                                        :animation-duration="2000" :size="50" color="#06C755" />
                                    <p class="text-center" v-if="generating"><strong>Generating...</strong></p>
                                    <div v-if="!generating && faces.length > 0" id="cubemap">
                                        <div id="container-output">
                                            <output id="faces">
                                                <a v-for="(face, index) in faces" :key="index"
                                                    :style="{ position: 'absolute', left: facePositions[face.faceName].x * 200 + 'px', top: facePositions[face.faceName].y * 200 + 'px' }"
                                                    :href="face.url" :download="face.download">
                                                    <img :src="face.preview" :title="face.faceName" />
                                                </a>
                                            </output>
                                        </div>
                                    </div>
                                    <button v-if="!generating && faces.length > 0" @click="uploadImages" type="submit"
                                        class="mt-4 btn-pers" id="login_button"><i class="fa-solid fa-paper-plane"></i>
                                        Add</button>
                                </section>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
import useEventBus from '@/composables/useEventBus'
import UserRequest from '@/restful/UserRequest';
const { emitEvent } = useEventBus();
import { ref } from 'vue';
import { FlowerSpinner } from 'epic-spinners';

export default {
    name: "MapAdd",
    components: {
        FlowerSpinner
    },
    data() {
        return {
            isTab: 'text',
            previewImageSrc: null,
            member: {
                name: null,
                email: null,
                line_user_id: null,
            },
            errors: {
                content_type: null,
                content_data_type: null,
                content_data_text: null,
                image_content: null,
            },
            dataText: {
                content_type: 'text',
                content_data: {
                    type: 'text',
                    text: null,
                },
            },
            dataImage: {
                content_type: 'image',
                image_content: null,
            },
            facePositions: {
                pz: { x: 1, y: 1 },
                nz: { x: 3, y: 1 },
                px: { x: 2, y: 1 },
                nx: { x: 0, y: 1 },
                py: { x: 1, y: 0 },
                ny: { x: 1, y: 2 }
            }
        }
    },
    computed: {
        orderedFaces() {
            // Sắp xếp các mặt theo vị trí x và y
            return this.faces.slice().sort((a, b) => {
                const posA = this.facePositions[a.faceName];
                const posB = this.facePositions[b.faceName];
                if (posA.y === posB.y) {
                    return posA.x - posB.x;
                }
                return posA.y - posB.y;
            });
        }
    },
    methods: {
        addMap: async function () {
            try {
                var dataSubmit = null;
                if (this.isTab == 'text') dataSubmit = this.dataText;
                if (this.isTab == 'image') {
                    dataSubmit = new FormData();
                    dataSubmit.append('content_type', 'image');
                    dataSubmit.append('image_content', this.dataImage.image_content);
                }
                const { messages } = await UserRequest.post('content/add', dataSubmit, true);
                emitEvent('eventSuccess', messages[0]);
                for (let key in this.errors) this.errors[key] = null;
                var closePW = window.document.getElementById('addMap');
                closePW.click();
                this.resetData();
                emitEvent('eventRegetMaps', '');
            }
            catch (error) {
                if (error.errors) this.errors = error.errors;
                else for (let key in this.errors) this.errors[key] = null;
                if (error.messages) emitEvent('eventError', error.messages[0]);
            }
        },
        resetData: function () {
            this.dataText = {
                content_type: 'text',
                content_data: {
                    type: 'text',
                    text: null,
                },
            };
            this.dataImage = {
                content_type: 'image',
                image_content: null,
            };
        },
        previewImage(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    this.previewImageSrc = e.target.result;
                    this.dataImage.image_content = file;
                };
                reader.readAsDataURL(file);
            } else this.removeFile();
            this.loadImage(event);
        },
        removeFile: function () {
            this.previewImageSrc = null;
            this.dataImage.image_content = null;
            this.$refs.fileInput.value = '';
        },
    },
    setup() {
        const rotation = ref(180);
        const interpolation = ref('lanczos');
        const format = ref('png');
        const generating = ref(false);
        const faces = ref([]);
        const submitAdress = ref('')
        const submitCoordinates = ref('');

        var fullfile;
        const loadImage = (event) => {
            const file = event.target.files[0];
            fullfile = file;

            if (!file) return;

            const img = new Image();
            img.src = URL.createObjectURL(file);
            img.onload = () => {
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
                const data = ctx.getImageData(0, 0, canvas.width, canvas.height);

                processImage(data);
            };
        };

        const processImage = (data) => {
            faces.value = [];
            generating.value = true;

            const facePositions = {
                pz: { x: 1, y: 1 },
                nz: { x: 3, y: 1 },
                px: { x: 2, y: 1 },
                nx: { x: 0, y: 1 },
                py: { x: 1, y: 0 },
                ny: { x: 1, y: 2 }
            };

            const options = {
                data,
                rotation: Math.PI * rotation.value / 180,
                interpolation: interpolation.value
            };

            Object.entries(facePositions).forEach(([faceName, position]) => {
                renderFace(data, faceName, position, options);
            });
        };

        const renderFace = (data, faceName, position, options) => {
            const worker = new Worker('convert.js');

            worker.onmessage = ({ data: imageData }) => {
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                canvas.width = imageData.width;
                canvas.height = imageData.height;
                ctx.putImageData(imageData, 0, 0);

                canvas.toBlob((blob) => {
                    const url = URL.createObjectURL(blob);
                    faces.value.push({
                        faceName,
                        preview: url,
                        download: `${faceName}.${format.value}`,
                        url
                    });

                    if (faces.value.length === 6) {
                        generating.value = false;
                    }
                }, `image/${format.value}`);
            };

            worker.postMessage({ ...options, face: faceName });
        };
        const uploadImages = () => {
            if (faces.value.length !== 6) {
                console.error("You must generate all 6 faces before uploading.");
                return;
            }
            const formData = new FormData();
            const promises = [];

            faces.value.forEach(face => {
                promises.push(
                    fetch(face.url)
                        .then(response => response.blob())
                        .then(blob => {
                            formData.append(face.faceName, blob);
                            console.log(blob);
                        })
                        .catch(error => {
                            console.error('Error fetching image:', error);
                        })
                );
            });
            console.log(formData);
            emitEvent('eventLoading', true);
            let closePW;
            Promise.all(promises)
                .then(() => {
                    formData.append('imagefull', fullfile);
                    formData.append('address', submitAdress.value);
                    formData.append('coordinates', submitCoordinates.value);

                    fetch('http://localhost:8000/api/map/add', {
                        method: 'POST',
                        body: formData
                    })
                        .then(response => {
                            if (!response.ok) {
                                throw new Error('Error uploading images.');
                            }
                            return response.json();
                        })
                        .then(data => {
                            emitEvent('eventLoading', false);
                            emitEvent('eventSuccess', "Images uploaded successfully !");
                            console.log(data);
                            closePW = window.document.getElementById('modal-add-map');
                            closePW.click();
                            emitEvent('eventRegetMaps', '');
                        })
                        .catch(error => {
                            console.error(error);
                            emitEvent("eventError", "Images uploaded fail !");
                        });
                });
        };
        return {
            rotation,
            interpolation,
            format,
            generating,
            faces,
            loadImage,
            uploadImages,
            submitAdress,
            submitCoordinates
        };
    },
    watch: {

    },
}
</script>

<style scoped>
#container-output {
    width: 100%;
    height: fit-content;
    display: flex;
    justify-content: center;
}

#faces {
    width: 800px;
    height: 600px;
    position: relative;
}

#faces img {
    width: 200px;
    height: 200px;
}

.loading-component {
    margin-top: 0px !important;
}

.loadContent {
    border: 1px solid;
    border-color: #dee2e6;
    margin-top: -1px;
    padding: 10px;
    border-bottom-left-radius: 6px;
    border-bottom-right-radius: 6px;
}

.colorImage {
    color: var(--blue-color);
}

.colorText {
    color: var(--brown-color)
}

.mainTab>li:hover {
    transition: all 0.5s;
}

.mainTab>li {
    transition: all 0.5s;
}

.modal.fade.show {
    padding-left: 0px;
}

.modal-content {
    margin-top: 100px;
    border-radius: 10px;
}


@import url('https://fonts.googleapis.com/css2?family=Reem+Kufi+Ink');

#big {
    display: flex;
    position: relative;
}

.btn-pers {
    position: relative;
    left: 50%;
    padding: 1em 2.5em;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    font-weight: 700;
    color: #000;
    background-color: #fff;
    border: none;
    border-radius: 45px;
    box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease 0s;
    cursor: pointer;
    outline: none;
    transform: translateX(-50%);
}

.btn-pers:hover {
    background-color: var(--user-color);
    box-shadow: 0px 15px 20px #80ffb5;
    color: #fff;
    transform: translate(-50%, -7px);
}

.btn-pers:active {
    transform: translate(-50%, -1px);
}

#inputPassword {
    padding-right: 26px;
}


.modal-dialog {
    max-width: 1000px;
}

.minAvatar {
    background-color: #e9ecef;
    position: relative;
    text-align: center;
    width: 100%;
    height: 320px;
    border-radius: 6px;
    cursor: pointer;
    display: flex;
    justify-content: center;
    align-items: center;
    transition: all 0.5s ease;
}

.minAvatar .preview {
    width: 100%;
    height: 300px;
    object-fit: cover;
    border-radius: 6px;
    cursor: default;
}

.minAvatar:hover {
    transition: all 0.5s ease;
    background: #E8F5E9;
}

.input-file {
    opacity: 0;
    top: 0px;
    left: 0px;
    position: absolute;
    cursor: pointer;
    width: 100%;
    height: 300px;
}

.box-preview {
    position: relative;
}

.iconClound {
    cursor: pointer;
    font-size: 60px;
    color: var(--user-color);
}

img.close {
    position: absolute;
    top: -6px;
    right: -6px;
    width: 16px;
}
</style>
