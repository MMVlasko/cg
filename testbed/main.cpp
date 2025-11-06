#include "veekay/input.hpp"
#include <format>
#include <climits>
#include <vector>
#include <stack>
#include <iostream>
#include <fstream>
#include <cmath>
#include <veekay/veekay.hpp>
#include <imgui.h>
#include <vulkan/vulkan_core.h>
#include <lodepng.h>

namespace {

constexpr uint32_t max_models = 1024;

struct Vertex {
	veekay::vec3 position;
	veekay::vec3 normal;
	veekay::vec2 uv;
};

struct SceneUniforms {
	veekay::mat4 view_projection;
};

struct alignas(16) ModelUniforms {
	veekay::mat4 model;
	veekay::vec3 albedo_color;
};

struct Mesh {
	veekay::graphics::Buffer* vertex_buffer;
	veekay::graphics::Buffer* edge_buffer;
	veekay::graphics::Buffer* index_buffer;
	uint32_t indices;
	uint32_t edge_indices;
};

struct Transform {
	veekay::vec3 position = {};
	veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
	veekay::vec3 rotation = {};

	[[nodiscard]] veekay::mat4 matrix() const;
};

struct Model {
	Mesh mesh;
	Transform transform;
	veekay::vec3 albedo_color;
	std::vector<Model*> children;
};

struct Camera {
	constexpr static float default_fov = 60.0f;
	constexpr static float default_near_plane = 0.01f;
	constexpr static float default_far_plane = 100.0f;

	veekay::vec3 position = {};
	veekay::vec3 rotation = {};

	float fov = default_fov;
	float near_plane = default_near_plane;
	float far_plane = default_far_plane;
	float speed = 1.;
	float scaling = 1.;
	float R = 4.;
	bool is_animation_frozen = false;
	bool reverse = false;

	[[nodiscard]] veekay::mat4 view() const;

	[[nodiscard]] veekay::mat4 view_projection(float aspect_ratio) const;
};

inline namespace {
	Camera camera{
		.position = {10.0f, 10.0f, 9.0f},
		.rotation = {54.0f, -139.0f, -26.0f}
	};

	std::vector<Model> models;
}

inline namespace {
	VkShaderModule vertex_shader_module;
	VkShaderModule fragment_shader_module;

	VkDescriptorPool descriptor_pool;
	VkDescriptorSetLayout descriptor_set_layout;
	VkDescriptorSet descriptor_set;

	VkPipelineLayout pipeline_layout;
	VkPipeline pipeline;
	VkPipeline wireframe_pipeline;

	veekay::graphics::Buffer* scene_uniforms_buffer;
	veekay::graphics::Buffer* model_uniforms_buffer;

	Mesh cube_mesh;
	Mesh pyramid_mesh;

	veekay::graphics::Texture* missing_texture;
	VkSampler missing_texture_sampler;

	veekay::graphics::Texture* texture;
	VkSampler texture_sampler;
}

float toRadians(float degrees) {
	return degrees * static_cast<float>(M_PI) / 180.0f;
}

veekay::mat4 Transform::matrix() const {
	const auto scaling_mtx = veekay::mat4::scaling(scale);
	const auto rot_mtx_x = veekay::mat4::rotation({1., .0, .0}, rotation.x);
	const auto rot_mtx_y = veekay::mat4::rotation({.0, -1., .0}, rotation.y);
	const auto rot_mtx_z = veekay::mat4::rotation({.0, .0, 1.}, rotation.z);
	auto t = veekay::mat4::translation(position);

	return scaling_mtx * rot_mtx_x * rot_mtx_y * rot_mtx_z * t;
}

veekay::mat4 Camera::view() const {
	const auto t = veekay::mat4::translation(-position);
	const auto rot_mtx_x = veekay::mat4::rotation({1., .0, .0}, toRadians(rotation.x));
	const auto rot_mtx_y = veekay::mat4::rotation({.0, -1., .0}, toRadians(rotation.y));
	const auto rot_mtx_z = veekay::mat4::rotation({.0, .0, 1.}, toRadians(rotation.z));

	return t * rot_mtx_x * rot_mtx_y * rot_mtx_z;
}

veekay::mat4 Camera::view_projection(float aspect_ratio) const {
	auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);

	return view() * projection;
}

VkShaderModule loadShaderModule(const char* path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	size_t size = file.tellg();
	std::vector<uint32_t> buffer(size / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(buffer.data()), size);
	file.close();

	VkShaderModuleCreateInfo info{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = buffer.data(),
	};

	VkShaderModule result;
	if (vkCreateShaderModule(veekay::app.vk_device, &
	                         info, nullptr, &result) != VK_SUCCESS) {
		return nullptr;
	}

	return result;
}

void initialize(VkCommandBuffer cmd) {
	VkDevice& device = veekay::app.vk_device;
	VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;

	{
		vertex_shader_module = loadShaderModule("../../shaders/shader.vert.spv");
		if (!vertex_shader_module) {
			std::cerr << "Failed to load Vulkan vertex shader from file\n";
			veekay::app.running = false;
			return;
		}

		fragment_shader_module = loadShaderModule("../../shaders/shader.frag.spv");
		if (!fragment_shader_module) {
			std::cerr << "Failed to load Vulkan fragment shader from file\n";
			veekay::app.running = false;
			return;
		}

		VkPipelineShaderStageCreateInfo stage_infos[2];

		stage_infos[0] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader_module,
			.pName = "main",
		};

		stage_infos[1] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader_module,
			.pName = "main",
		};

		VkVertexInputBindingDescription buffer_binding{
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		};

		VkVertexInputAttributeDescription attributes[] = {
			{
				.location = 0,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, position),
			},
			{
				.location = 1,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, normal),
			},
			{
				.location = 2,
				.binding = 0,
				.format = VK_FORMAT_R32G32_SFLOAT,
				.offset = offsetof(Vertex, uv),
			}
		};

		VkPipelineVertexInputStateCreateInfo input_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &buffer_binding,
			.vertexAttributeDescriptionCount = std::size(attributes),
			.pVertexAttributeDescriptions = attributes,
		};

		VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		};

		VkPipelineRasterizationStateCreateInfo raster_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_BACK_BIT,
			.frontFace = VK_FRONT_FACE_CLOCKWISE,
			.lineWidth = 1.0f,
		};

		VkPipelineMultisampleStateCreateInfo sample_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			.sampleShadingEnable = false,
			.minSampleShading = 1.0f,
		};

		VkViewport viewport{
			.x = 0.0f,
			.y = 0.0f,
			.width = static_cast<float>(veekay::app.window_width),
			.height = static_cast<float>(veekay::app.window_height),
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};

		VkRect2D scissor{
			.offset = {0, 0},
			.extent = {veekay::app.window_width, veekay::app.window_height},
		};

		VkPipelineViewportStateCreateInfo viewport_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,

			.viewportCount = 1,
			.pViewports = &viewport,

			.scissorCount = 1,
			.pScissors = &scissor,
		};

		VkPipelineDepthStencilStateCreateInfo depth_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = true,
			.depthWriteEnable = true,
			.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
		};

		VkPipelineColorBlendAttachmentState attachment_info{
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
			                  VK_COLOR_COMPONENT_G_BIT |
			                  VK_COLOR_COMPONENT_B_BIT |
			                  VK_COLOR_COMPONENT_A_BIT,
		};

		VkPipelineColorBlendAttachmentState wireframe_attachment{
			.blendEnable = VK_TRUE,
			.srcColorBlendFactor = VK_BLEND_FACTOR_CONSTANT_COLOR,
			.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
			.colorBlendOp = VK_BLEND_OP_ADD,
			.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
			.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
			.alphaBlendOp = VK_BLEND_OP_ADD,
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
							  VK_COLOR_COMPONENT_G_BIT |
							  VK_COLOR_COMPONENT_B_BIT |
							  VK_COLOR_COMPONENT_A_BIT,
		};

		VkPipelineColorBlendStateCreateInfo blend_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,

			.logicOpEnable = false,
			.logicOp = VK_LOGIC_OP_COPY,

			.attachmentCount = 1,
			.pAttachments = &attachment_info
		};

		VkPipelineColorBlendStateCreateInfo wireframe_blend_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
			.logicOpEnable = VK_FALSE,
			.attachmentCount = 1,
			.pAttachments = &wireframe_attachment,
		};

		{
			VkDescriptorPoolSize pools[] = {
				{
					.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 8,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.descriptorCount = 8,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 8,
				}
			};

			VkDescriptorPoolCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
				.maxSets = 1,
				.poolSizeCount = std::size(pools),
				.pPoolSizes = pools,
			};

			if (vkCreateDescriptorPool(device, &info, nullptr,
			                           &descriptor_pool) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor pool\n";
				veekay::app.running = false;
				return;
			}
		}

		{
			VkDescriptorSetLayoutBinding bindings[] = {
				{
					.binding = 0,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
			};

			VkDescriptorSetLayoutCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
				.bindingCount = std::size(bindings),
				.pBindings = bindings,
			};

			if (vkCreateDescriptorSetLayout(device, &info, nullptr,
			                                &descriptor_set_layout) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor set layout\n";
				veekay::app.running = false;
				return;
			}
		}

		{
			VkDescriptorSetAllocateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &descriptor_set_layout,
			};

			if (vkAllocateDescriptorSets(device, &info, &descriptor_set) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor set\n";
				veekay::app.running = false;
				return;
			}
		}

		VkPipelineLayoutCreateInfo layout_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &descriptor_set_layout,
		};

		if (vkCreatePipelineLayout(device, &layout_info,
		                           nullptr, &pipeline_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline layout\n";
			veekay::app.running = false;
			return;
		}

		VkGraphicsPipelineCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = 2,
			.pStages = stage_infos,
			.pVertexInputState = &input_state_info,
			.pInputAssemblyState = &assembly_state_info,
			.pViewportState = &viewport_info,
			.pRasterizationState = &raster_info,
			.pMultisampleState = &sample_info,
			.pDepthStencilState = &depth_info,
			.pColorBlendState = &blend_info,
			.layout = pipeline_layout,
			.renderPass = veekay::app.vk_render_pass,
		};

		if (vkCreateGraphicsPipelines(device, nullptr,
		                              1, &info, nullptr, &pipeline) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline\n";
			veekay::app.running = false;
			return;
		}

		VkPipelineInputAssemblyStateCreateInfo line_assembly_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
		};

		VkPipelineRasterizationStateCreateInfo line_raster_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.polygonMode = VK_POLYGON_MODE_LINE,
			.frontFace = VK_FRONT_FACE_CLOCKWISE,
			.lineWidth = 1.5f,
		};

		VkGraphicsPipelineCreateInfo line_pipeline_info = info;
		line_pipeline_info.pInputAssemblyState = &line_assembly_info;
		line_pipeline_info.pRasterizationState = &line_raster_info;
		line_pipeline_info.pColorBlendState = &wireframe_blend_info;

		if (vkCreateGraphicsPipelines(device, nullptr, 1, &line_pipeline_info, nullptr, &wireframe_pipeline) != VK_SUCCESS) {
			std::cerr << "Failed to create wireframe pipeline\n";
			veekay::app.running = false;
			return;
		}

	}

	scene_uniforms_buffer = new veekay::graphics::Buffer(
		sizeof(SceneUniforms),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	model_uniforms_buffer = new veekay::graphics::Buffer(
		max_models * sizeof(ModelUniforms),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	{
		VkSamplerCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		};

		if (vkCreateSampler(device, &info, nullptr, &missing_texture_sampler) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan texture sampler\n";
			veekay::app.running = false;
			return;
		}

		uint32_t pixels[] = {
			0xff000000, 0xffff00ff,
			0xffff00ff, 0xff000000,
		};

		missing_texture = new veekay::graphics::Texture(cmd, 2, 2,
		                                                VK_FORMAT_B8G8R8A8_UNORM,
		                                                pixels);
	}

	{
		VkDescriptorBufferInfo buffer_infos[] = {
			{
				.buffer = scene_uniforms_buffer->buffer,
				.offset = 0,
				.range = sizeof(SceneUniforms),
			},
			{
				.buffer = model_uniforms_buffer->buffer,
				.offset = 0,
				.range = sizeof(ModelUniforms),
			},
		};

		VkWriteDescriptorSet write_infos[] = {
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo = &buffer_infos[0],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 1,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
				.pBufferInfo = &buffer_infos[1],
			},
		};

		vkUpdateDescriptorSets(device, std::size(write_infos),
		                       write_infos, 0, nullptr);
	}

	{
	std::vector<Vertex> vertices = {
		{{-1.0f, -1.0f, 0.0f}, {}, {0.5f, 1.0f}},
		{{1.0f, -1.0f, 0.0f}, {}, {1.0f, 0.5f}},
		{{-1.0f, 1.0f, 0.0f}, {}, {0.5f, 0.5f}},
		{{1.0f, 1.0f, 0.0f}, {}, {0.0f, 0.5f}},
		{{-1.0f, -1.0f, 2.0f}, {}, {0.5f, 1.0f}},
		{{1.0f, -1.0f, 2.0f}, {}, {1.0f, 0.5f}},
		{{-1.0f, 1.0f, 2.0f}, {}, {0.5f, 0.5f}},
		{{1.0f, 1.0f, 2.0f}, {}, {0.0f, 0.5f}}
	};

	std::vector<uint32_t> indices = {
		0, 1, 3,
		0, 3, 2,
		0, 4, 1,
		4, 7, 5,
		4, 6, 7,
		1, 5, 7,
		1, 7, 3,
		3, 7, 6,
		3, 6, 2,
		2, 6, 4,
		2, 4, 0,
		0, 4, 5,
		0, 5, 1
	};

	std::vector<uint32_t> edge_indices = {
		0, 1, 1, 3, 3, 2, 2, 0,
		4, 5, 5, 7, 7, 6, 6, 4,
		0, 4, 1, 5, 2, 6, 3, 7
	};

	cube_mesh.vertex_buffer = new veekay::graphics::Buffer(
		vertices.size() * sizeof(Vertex), vertices.data(),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

	cube_mesh.index_buffer = new veekay::graphics::Buffer(
		indices.size() * sizeof(uint32_t), indices.data(),
		VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

	cube_mesh.edge_buffer = new veekay::graphics::Buffer(
		edge_indices.size() * sizeof(uint32_t), edge_indices.data(),
		VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

	cube_mesh.indices = static_cast<uint32_t>(indices.size());
	cube_mesh.edge_indices = static_cast<uint32_t>(edge_indices.size());
}

{
	std::vector<Vertex> vertices = {
		{{-1.0f, -1.0f, 0.0f}, {}, {0.5f, 1.0f}},
		{{1.0f, -1.0f, 0.0f}, {}, {1.0f, 0.5f}},
		{{-1.0f, 1.0f, 0.0f}, {}, {0.5f, 0.5f}},
		{{1.0f, 1.0f, 0.0f}, {}, {0.0f, 0.5f}},
		{{0.0f, 0.0f, 4.0f}, {}, {0.5f, 0.0f}}
	};

	std::vector<uint32_t> indices = {
		0, 1, 3,
		0, 3, 2,
		0, 4, 1,
		2, 4, 0,
		1, 4, 3,
		3, 4, 2
	};

	std::vector<uint32_t> edge_indices = {
		0, 1, 1, 3, 3, 2, 2, 0,
		0, 4, 1, 4, 2, 4, 3, 4
	};

	pyramid_mesh.vertex_buffer = new veekay::graphics::Buffer(
		vertices.size() * sizeof(Vertex), vertices.data(),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

	pyramid_mesh.index_buffer = new veekay::graphics::Buffer(
		indices.size() * sizeof(uint32_t), indices.data(),
		VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

	pyramid_mesh.edge_buffer = new veekay::graphics::Buffer(
		edge_indices.size() * sizeof(uint32_t), edge_indices.data(),
		VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

	pyramid_mesh.indices = static_cast<uint32_t>(indices.size());
	pyramid_mesh.edge_indices = static_cast<uint32_t>(edge_indices.size());
}


	models.emplace_back(Model{
		.mesh = pyramid_mesh,
		.transform = Transform{},
		.albedo_color = veekay::vec3{3.0f, 0.0f, 0.0f}
	});

	models.emplace_back(Model{
		.mesh = pyramid_mesh,
		.transform = Transform{
			.position = {0.0f, 0.0f, 7.0f},
			.scale = {0.3f, 0.3f, 0.3f}
		},
		.albedo_color = veekay::vec3{3.0f, 0.0f, 0.0f}
	});

	models.emplace_back(Model{
		.mesh = cube_mesh,
		.transform = Transform{
			.position = {5.0f, 0.0f, 0.0f},
			.scale = {0.5f, 0.5f, 0.5f}
		},
		.albedo_color = veekay::vec3{3.0f, 4.0f, 0.0f}
	});

	models[0].children = {&models[1], &models[2]};
}

void shutdown() {
	VkDevice& device = veekay::app.vk_device;

	vkDestroySampler(device, missing_texture_sampler, nullptr);
	delete missing_texture;

	delete pyramid_mesh.index_buffer;
	delete pyramid_mesh.vertex_buffer;

	delete cube_mesh.index_buffer;
	delete cube_mesh.vertex_buffer;

	delete model_uniforms_buffer;
	delete scene_uniforms_buffer;

	vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
	vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

	vkDestroyPipeline(device, pipeline, nullptr);
	vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
	vkDestroyShaderModule(device, fragment_shader_module, nullptr);
	vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}

void hierarchical_transformation(const Model& model, std::stack<veekay::mat4>& matrix_stack, std::vector<ModelUniforms>& model_uniforms, size_t& index) {
	auto matrix = model.transform.matrix() * matrix_stack.top();
	matrix_stack.push(matrix);
	model_uniforms[index] = ModelUniforms(matrix, model.albedo_color);
	index++;
	for (auto& child : model.children) {
		hierarchical_transformation(*child, matrix_stack, model_uniforms, index);
	}
	matrix_stack.pop();
}

void update(double time) {
	ImGui::Begin("Controls:");
	ImGui::SliderFloat("Orbit radius", &camera.R, 3.f, 35.f);

	if (ImGui::SliderFloat("Scaling", &camera.scaling, 0.1f, 6.0f)) {
		models[0].transform.scale.x = camera.scaling;
		models[0].transform.scale.y = camera.scaling;
		models[0].transform.scale.z = camera.scaling;
	}

	ImGui::SliderFloat("Animation speed", &camera.speed, 0.01, 5);

	if (ImGui::Button("Reset camera")) {
		camera.rotation = {54.0f, -139.0f, -26.0f};
		camera.position = {10.0f, 10.0f, 12.0f};
	}
	if (ImGui::Button("Pause animation")) {
		camera.is_animation_frozen ^= 1;
	}

	if (ImGui::Button("Reverse animation")) {
		camera.reverse ^= 1;
	}

	ImGui::End();


	if (!camera.is_animation_frozen) {
		float theta = camera.speed * time * (camera.reverse ? -1.0f : 1.0f);

		models[0].transform.position.x = camera.R * cosf(theta);
		models[0].transform.position.y = camera.R * sinf(theta);

		models[0].transform.rotation.z = theta + M_PI / 4;
		models[2].transform.position.x = 3 * cosf(theta * 2.5);
		models[2].transform.position.y = 3 * sinf(theta * 2.5);
	}

	if (!ImGui::IsWindowHovered()) {
		using namespace veekay::input;

		if (mouse::isButtonDown(mouse::Button::left)) {
			auto move_delta = mouse::cursorDelta();

			const float rotate_x = -90 * move_delta.y / veekay::app.window_height;
			const float rotate_y = 90 * move_delta.x / veekay::app.window_width;
			camera.rotation.x += rotate_x;
			camera.rotation.y += rotate_y;
		}
		auto view_t = veekay::mat4::transpose(camera.view());

		veekay::vec3 right = veekay::vec3::normalized({view_t[0][0], view_t[0][1], view_t[0][2]});
		veekay::vec3 up = veekay::vec3::normalized({-view_t[1][0], -view_t[1][1], -view_t[1][2]});
		veekay::vec3 front =veekay::vec3::normalized({view_t[2][0], view_t[2][1], view_t[2][2]});

		if (keyboard::isKeyDown(keyboard::Key::equal))
			camera.position += front * 0.1f;

		if (keyboard::isKeyDown(keyboard::Key::minus))
			camera.position -= front * 0.1f;

		if (keyboard::isKeyDown(keyboard::Key::right))
			camera.position += right * 0.1f;

		if (keyboard::isKeyDown(keyboard::Key::left))
			camera.position -= right * 0.1f;

		if (keyboard::isKeyDown(keyboard::Key::up))
			camera.position += up * 0.1f;

		if (keyboard::isKeyDown(keyboard::Key::down))
			camera.position -= up * 0.1f;
	}

	float aspect_ratio = static_cast<float>(veekay::app.window_width) / static_cast<float>(veekay::app.window_height);
	SceneUniforms scene_uniforms{
		.view_projection = camera.view_projection(aspect_ratio),
	};



	std::vector<ModelUniforms> model_uniforms(models.size());
	std::stack<veekay::mat4> matrix_stack;
	matrix_stack.push(veekay::mat4::identity());
	size_t index = 0;
    hierarchical_transformation(models[0], matrix_stack, model_uniforms, index);

	*static_cast<SceneUniforms *>(scene_uniforms_buffer->mapped_region) = scene_uniforms;
	std::ranges::copy(model_uniforms,
	                  static_cast<ModelUniforms*>(model_uniforms_buffer->mapped_region));
}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
	vkResetCommandBuffer(cmd, 0);

	VkCommandBufferBeginInfo begin_info{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(cmd, &begin_info);

	constexpr VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
	constexpr VkClearValue clear_depth{.depthStencil = {1.0f, 0}};
	VkClearValue clear_values[] = {clear_color, clear_depth};

	VkRenderPassBeginInfo pass_info{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = veekay::app.vk_render_pass,
		.framebuffer = framebuffer,
		.renderArea = {.extent = {veekay::app.window_width, veekay::app.window_height}},
		.clearValueCount = 2,
		.pClearValues = clear_values,
	};

	vkCmdBeginRenderPass(cmd, &pass_info, VK_SUBPASS_CONTENTS_INLINE);

	VkDeviceSize zero_offset = 0;
	VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
	VkBuffer current_index_buffer = VK_NULL_HANDLE;

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	for (size_t i = 0, n = models.size(); i < n; ++i) {
		const Model& model = models[i];
		const Mesh& mesh = model.mesh;

		if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
			current_vertex_buffer = mesh.vertex_buffer->buffer;
			vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
		}
		if (current_index_buffer != mesh.index_buffer->buffer) {
			current_index_buffer = mesh.index_buffer->buffer;
			vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
		}

		uint32_t offset = i * sizeof(ModelUniforms);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &descriptor_set, 1, &offset);
		vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
	}

	float edge_color[4] = {0.0f, 0.0f, 1.0f, 1.0f};
	vkCmdSetBlendConstants(cmd, edge_color);
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, wireframe_pipeline);
	for (size_t i = 0, n = models.size(); i < n; ++i) {
		const Model& model = models[i];
		const Mesh& mesh = model.mesh;

		if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
			current_vertex_buffer = mesh.vertex_buffer->buffer;
			vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
		}
		if (current_index_buffer != mesh.edge_buffer->buffer) {
			current_index_buffer = mesh.edge_buffer->buffer;
			vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
		}

		uint32_t offset = i * sizeof(ModelUniforms);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &descriptor_set, 1, &offset);
		vkCmdDrawIndexed(cmd, mesh.edge_indices, 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}


}

int main() {
	return veekay::run({
		.init = initialize,
		.shutdown = shutdown,
		.update = update,
		.render = render,
	});
}