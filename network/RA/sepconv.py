import cupy
import torch
import re
import numpy as np
from torch.autograd import Variable as var
import torch.nn as nn
import IPython
class SATDLoss(nn.Module):
	def __init__(self):
		super(SATDLoss, self).__init__()
		self.r1 = np.array([
			[1, 0, 0, 0, 1, 0, 0, 0, ],
			[0, 1, 0, 0, 0, 1, 0, 0, ],
			[0, 0, 1, 0, 0, 0, 1, 0, ],
			[0, 0, 0, 1, 0, 0, 0, 1, ],
			[1, 0, 0, 0, -1, 0, 0, 0, ],
			[0, 1, 0, 0, 0, -1, 0, 0, ],
			[0, 0, 1, 0, 0, 0, -1, 0, ],
			[0, 0, 0, 1, 0, 0, 0, -1, ],
		], dtype=np.float32)
		self.r2 = np.array([
			[1, 0, 1, 0, 0, 0, 0, 0, ],
			[0, 1, 0, 1, 0, 0, 0, 0, ],
			[1, 0, -1, 0, 0, 0, 0, 0, ],
			[0, 1, 0, -1, 0, 0, 0, 0, ],
			[0, 0, 0, 0, 1, 0, 1, 0, ],
			[0, 0, 0, 0, 0, 1, 0, 1, ],
			[0, 0, 0, 0, 1, 0, -1, 0, ],
			[0, 0, 0, 0, 0, 1, 0, -1, ],
		], dtype=np.float32)
		self.r3 = np.array([
			[1, 1, 0, 0, 0, 0, 0, 0, ],
			[1, -1, 0, 0, 0, 0, 0, 0, ],
			[0, 0, 1, 1, 0, 0, 0, 0, ],
			[0, 0, 1, -1, 0, 0, 0, 0, ],
			[0, 0, 0, 0, 1, 1, 0, 0, ],
			[0, 0, 0, 0, 1, -1, 0, 0, ],
			[0, 0, 0, 0, 0, 0, 1, 1, ],
			[0, 0, 0, 0, 0, 0, 1, -1, ],
		], dtype=np.float32)

		self.l1 = self.r1.transpose()
		self.l2 = self.r2.transpose()
		self.l3 = self.r3.transpose()

		self.lmul = var(torch.from_numpy(self.l1.dot(self.l2).dot(self.l3))).cuda()
		self.rmul = var(torch.from_numpy(self.r1.dot(self.r2).dot(self.r3))).cuda()

	def forward(self, input, label):
		intSample = input.size(0)
		intInputDepth = input.size(1)
		intInputHeight = input.size(2)
		intInputWidth = input.size(3)

		diff = input - label
		difft = diff.new(intSample,intInputDepth,intInputHeight,intInputWidth).zero_()

		for i in range(0,intInputHeight,8):
			for j in range(0,intInputWidth,8):
				difft[:,:,i:i+8,j:j+8] = self.lmul.matmul(diff[:,:,i:i+8,j:j+8]).matmul(self.rmul)
		difft = difft.abs()
		loss = torch.sum(difft)/torch.numel(difft)
		return loss

class FlowLoss(nn.Module):
	def __init__(self):
		super(FlowLoss, self).__init__()
		self.Gx = np.array([
			[-1,0,1, ],
			[-2,0,2, ],
			[-1,0,1, ],
		], dtype=np.float32)

		self.Gy = np.array([
			[1, 2, 1, ],
			[0, 0, 0, ],
			[-1, -2, -1, ],
		], dtype=np.float32)

		self.ConvGx = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False,groups=3)
		self.ConvGx.weight=nn.Parameter(torch.from_numpy(self.Gx).float().unsqueeze(0).unsqueeze(0).repeat(3,1,1,1))
		for param in self.ConvGx.parameters():
			param.requires_grad = False

		self.ConvGy = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False,groups=3)
		self.ConvGy.weight = nn.Parameter(torch.from_numpy(self.Gy).float().unsqueeze(0).unsqueeze(0).repeat(3,1,1,1))
		for param in self.ConvGy.parameters():
			param.requires_grad = False

	def forward(self, input, label, refL, refR):
		intSample = input.size(0)
		intInputDepth = input.size(1)
		intInputHeight = input.size(2)
		intInputWidth = input.size(3)

		diffGx = torch.abs(self.ConvGx(input)-self.ConvGx(label))
		diffGy = torch.abs(self.ConvGy(input) - self.ConvGy(label))
		diff = torch.cat([diffGx, diffGy], 1)

		loss = torch.sum(diff)/torch.numel(diff)
		return loss

kernel_Sepconv_updateOutput = '''
	extern "C" __global__ void kernel_Sepconv_updateOutput(
		const int n,
		const float* input,
		const float* vertical,
		const float* horizontal,
		float* output
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
		float dblOutput = 0.0;
		const int intSample = ( intIndex / SIZE_3(output) / SIZE_2(output) / SIZE_1(output) ) % SIZE_0(output);
		const int intDepth  = ( intIndex / SIZE_3(output) / SIZE_2(output)                  ) % SIZE_1(output);
		const int intY      = ( intIndex / SIZE_3(output)                                   ) % SIZE_2(output);
		const int intX      = ( intIndex                                                    ) % SIZE_3(output);
		for (int intFilterY = 0; intFilterY < SIZE_1(vertical); intFilterY += 1) {
			for (int intFilterX = 0; intFilterX < SIZE_1(horizontal); intFilterX += 1) {
				dblOutput += VALUE_4(input, intSample, intDepth, intY + intFilterY, intX + intFilterX) * VALUE_4(vertical, intSample, intFilterY, intY, intX) * VALUE_4(horizontal, intSample, intFilterX, intY, intX);
			}
		}
		output[intIndex] = dblOutput;
	} }
'''

kernel_Sepconv_updateVerticalGrad = '''
	extern "C" __global__ void kernel_Sepconv_updateVerticalGrad(
		const int n,
		const float* gradinput,
		const float* input,
		const float* horizontal,
		float* VerticalGrad
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
		float dblOutput = 0.0;
		const int intSample = ( intIndex / SIZE_3(VerticalGrad) / SIZE_2(VerticalGrad) / SIZE_1(VerticalGrad) ) % SIZE_0(VerticalGrad);
		const int intFilterY  = ( intIndex / SIZE_3(VerticalGrad) / SIZE_2(VerticalGrad)                  ) % SIZE_1(VerticalGrad);
		const int intY      = ( intIndex / SIZE_3(VerticalGrad)                                   ) % SIZE_2(VerticalGrad);
		const int intX      = ( intIndex                                                    ) % SIZE_3(VerticalGrad);
		for (int intFilterX = 0; intFilterX < SIZE_1(horizontal); intFilterX += 1) {
			dblOutput += VALUE_4(gradinput, intSample, 0, intY, intX)* VALUE_4(input, intSample, 0, intY+intFilterY, intX + intFilterX) * 
			VALUE_4(horizontal, intSample, intFilterX, intY, intX) +
			VALUE_4(gradinput, intSample, 1, intY, intX)* VALUE_4(input, intSample, 1, intY+intFilterY, intX + intFilterX) * 
			VALUE_4(horizontal, intSample, intFilterX, intY, intX) +
			VALUE_4(gradinput, intSample, 2, intY, intX)* VALUE_4(input, intSample, 2, intY+intFilterY, intX + intFilterX) * 
			VALUE_4(horizontal, intSample, intFilterX, intY, intX);
		}
		VerticalGrad[intIndex] = dblOutput;
	} }
'''

kernel_Sepconv_updateHorizontalGrad = '''
	extern "C" __global__ void kernel_Sepconv_updateHorizontalGrad(
		const int n,
		const float* gradinput,
		const float* input,
		const float* vertical,
		float* HorizontalGrad
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
		float dblOutput = 0.0;
		const int intSample = ( intIndex / SIZE_3(HorizontalGrad) / SIZE_2(HorizontalGrad) / SIZE_1(HorizontalGrad) ) % SIZE_0(HorizontalGrad);
		const int intFilterX  = ( intIndex / SIZE_3(HorizontalGrad) / SIZE_2(HorizontalGrad)                  ) % SIZE_1(HorizontalGrad);
		const int intY      = ( intIndex / SIZE_3(HorizontalGrad)                                   ) % SIZE_2(HorizontalGrad);
		const int intX      = ( intIndex                                                    ) % SIZE_3(HorizontalGrad);
		for (int intFilterY = 0; intFilterY < SIZE_1(vertical); intFilterY += 1) {
			dblOutput += VALUE_4(gradinput, intSample, 0, intY, intX)* VALUE_4(input, intSample, 0, intY+intFilterY, intX + intFilterX) * 
			VALUE_4(vertical, intSample, intFilterY, intY, intX) +
			VALUE_4(gradinput, intSample, 1, intY, intX)* VALUE_4(input, intSample, 1, intY+intFilterY, intX + intFilterX) * 
			VALUE_4(vertical, intSample, intFilterY, intY, intX) +
			VALUE_4(gradinput, intSample, 2, intY, intX)* VALUE_4(input, intSample, 2, intY+intFilterY, intX + intFilterX) * 
			VALUE_4(vertical, intSample, intFilterY, intY, intX);
		}
		HorizontalGrad[intIndex] = dblOutput;
	} }
'''

def cupy_kernel(strFunction, objectVariables):
	strKernel = globals()[strFunction]

	while True:
		objectMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

		if objectMatch is None:
			break
		# end

		intArg = int(objectMatch.group(2))

		strTensor = objectMatch.group(4)
		intSizes = objectVariables[strTensor].size()

		strKernel = strKernel.replace(objectMatch.group(), str(intSizes[intArg]))
	# end

	while True:
		objectMatch = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))', strKernel)

		if objectMatch is None:
			break
		# end

		intArgs = int(objectMatch.group(2))
		strArgs = objectMatch.group(4).split(',')

		strTensor = strArgs[0]
		intStrides = objectVariables[strTensor].stride()
		strIndex = [ '((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg]) + ')' for intArg in range(intArgs) ]

		strKernel = strKernel.replace(objectMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')
	# end

	return strKernel
# end

@cupy.util.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
	return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)
# end

class FunctionSepconv(torch.autograd.Function):
	def __init__(self):
		super(FunctionSepconv, self).__init__()
	# end

	def forward(self, input, vertical, horizontal):
		self.save_for_backward(input, vertical, horizontal)

		intSample = input.size(0)
		intInputDepth = input.size(1)
		intInputHeight = input.size(2)
		intInputWidth = input.size(3)
		intFilterSize = min(vertical.size(1), horizontal.size(1))
		intOutputHeight = min(vertical.size(2), horizontal.size(2))
		intOutputWidth = min(vertical.size(3), horizontal.size(3))

		assert(intInputHeight - intFilterSize == intOutputHeight - 1)
		assert(intInputWidth - intFilterSize == intOutputWidth - 1)

		assert(input.is_contiguous() == True)
		assert(vertical.is_contiguous() == True)
		assert(horizontal.is_contiguous() == True)

		output = input.new(intSample, intInputDepth, intOutputHeight, intOutputWidth).zero_()

		if input.is_cuda == True:
			class Stream:
				ptr = torch.cuda.current_stream().cuda_stream
			# end

			n = output.nelement()
			cupy_launch('kernel_Sepconv_updateOutput', cupy_kernel('kernel_Sepconv_updateOutput', {
				'input': input,
				'vertical': vertical,
				'horizontal': horizontal,
				'output': output
			}))(
				grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
				block=tuple([ 512, 1, 1 ]),
				args=[ n, input.data_ptr(), vertical.data_ptr(), horizontal.data_ptr(), output.data_ptr() ],
				stream=Stream
			)
		elif first.is_cuda == False:
			raise NotImplementedError()

		# end

		return output
	# end

	def backward(self, gradInput):
		input, vertical, horizontal = self.saved_tensors

		intSample = input.size(0)
		intInputDepth = input.size(1)
		intInputHeight = input.size(2)
		intInputWidth = input.size(3)
		intFilterSize = min(vertical.size(1), horizontal.size(1))
		intOutputHeight = min(vertical.size(2), horizontal.size(2))
		intOutputWidth = min(vertical.size(3), horizontal.size(3))

		assert(intInputHeight - intFilterSize == intOutputHeight - 1)
		assert(intInputWidth - intFilterSize == intOutputWidth - 1)
		gradInput=gradInput.contiguous()
		assert(gradInput.is_contiguous() == True)

		InputGrad = input.new(intSample, intInputDepth, intInputHeight, intInputWidth).zero_()
		VerticalGrad = input.new(intSample, intFilterSize, intOutputHeight, intOutputWidth).zero_()
		HorizontalGrad = input.new(intSample, intFilterSize, intOutputHeight, intOutputWidth).zero_()

		if gradInput.is_cuda == True:
			class Stream:
				ptr = torch.cuda.current_stream().cuda_stream

			n = VerticalGrad.nelement()
			cupy_launch('kernel_Sepconv_updateVerticalGrad', cupy_kernel('kernel_Sepconv_updateVerticalGrad', {
				'gradinput': gradInput,
				'input': input,
				'horizontal': horizontal,
				'VerticalGrad': VerticalGrad
			}))(
				grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
				block=tuple([512, 1, 1]),
				args=[n, gradInput.data_ptr(), input.data_ptr(), horizontal.data_ptr(), VerticalGrad.data_ptr()],
				stream=Stream
			)
			n = HorizontalGrad.nelement()
			cupy_launch('kernel_Sepconv_updateHorizontalGrad', cupy_kernel('kernel_Sepconv_updateHorizontalGrad', {
				'gradinput': gradInput,
				'input': input,
				'vertical': vertical,
				'HorizontalGrad': HorizontalGrad
			}))(
				grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
				block=tuple([512, 1, 1]),
				args=[n, gradInput.data_ptr(), input.data_ptr(), vertical.data_ptr(), HorizontalGrad.data_ptr()],
				stream=Stream
			)

		return InputGrad, VerticalGrad, HorizontalGrad
	# end
# end

def sepconvf(input, vertical, horizontal):
	intSample = input.size(0)
	intInputDepth = input.size(1)
	intInputHeight = input.size(2)
	intInputWidth = input.size(3)
	intFilterSize = min(vertical.size(1), horizontal.size(1))
	intOutputHeight = min(vertical.size(2), horizontal.size(2))
	intOutputWidth = min(vertical.size(3), horizontal.size(3))

	assert (intInputHeight - intFilterSize == intOutputHeight - 1)
	assert (intInputWidth - intFilterSize == intOutputWidth - 1)

	assert (input.is_contiguous() == True)
	assert (vertical.is_contiguous() == True)
	assert (horizontal.is_contiguous() == True)

	output = input.new(intSample, intInputDepth, intOutputHeight, intOutputWidth).zero_()

	filter_r = int(intFilterSize / 2)
	_, _, rows, cols = output.shape
	for row in range(rows):
		# print(row)
		for col in range(cols):
			filter_mat = vertical[0, :, row, col].repeat(1, 1).t().matmul(horizontal[0, :, row, col].repeat(1, 1))
			output[0, :, row, col] = torch.sum(
				torch.sum(input[0, :, row:row + 2 * filter_r + 1, col:col + 2 * filter_r + 1] * filter_mat, 1), 1)
	return output

class FunctionSepconv_cpu(torch.autograd.Function):
	def __init__(self):
		super(FunctionSepconv_cpu, self).__init__()
	# end

	def forward(self, input, vertical, horizontal):
		self.save_for_backward(input, vertical, horizontal)

		intSample = input.size(0)
		intInputDepth = input.size(1)
		intInputHeight = input.size(2)
		intInputWidth = input.size(3)
		intFilterSize = min(vertical.size(1), horizontal.size(1))
		intOutputHeight = min(vertical.size(2), horizontal.size(2))
		intOutputWidth = min(vertical.size(3), horizontal.size(3))

		assert(intInputHeight - intFilterSize == intOutputHeight - 1)
		assert(intInputWidth - intFilterSize == intOutputWidth - 1)

		assert(input.is_contiguous() == True)
		assert(vertical.is_contiguous() == True)
		assert(horizontal.is_contiguous() == True)

		output = input.new(intSample, intInputDepth, intOutputHeight, intOutputWidth).zero_()

		filter_r = int(intFilterSize/2)
		_, _, rows, cols = output.shape
		for row in range(rows):
			print(row)
			for col in range(cols):
				filter_mat = vertical[0,:,row,col].repeat(1,1).t().matmul(horizontal[0,:,row,col].repeat(1,1))
				output[0, :, row, col] = torch.sum(torch.sum(input[0,:,row :row + 2*filter_r +1,col :col + 2*filter_r+1]*filter_mat,1),1)


		return output
	# end

	def backward(self, gradOutput):
		input, vertical, horizontal = self.saved_tensors

		intSample = input.size(0)
		intInputDepth = input.size(1)
		intInputHeight = input.size(2)
		intInputWidth = input.size(3)
		intFilterSize = min(vertical.size(1), horizontal.size(1))
		intOutputHeight = min(vertical.size(2), horizontal.size(2))
		intOutputWidth = min(vertical.size(3), horizontal.size(3))

		assert(intInputHeight - intFilterSize == intOutputHeight - 1)
		assert(intInputWidth - intFilterSize == intOutputWidth - 1)

		assert(gradOutput.is_contiguous() == True)

		gradInput = input.new(intSample, intInputDepth, intInputHeight, intInputWidth).zero_() if self.needs_input_grad[0] == True else None
		gradVertical = input.new(intSample, intFilterSize, intOutputHeight, intOutputWidth).zero_() if self.needs_input_grad[1] == True else None
		gradHorizontal = input.new(intSample, intFilterSize, intOutputHeight, intOutputWidth).zero_() if self.needs_input_grad[2] == True else None


		return gradInput, gradVertical, gradHorizontal
	# end
# end

class ModuleSepconv(torch.nn.Module):
	def __init__(self):
		super(ModuleSepconv, self).__init__()
	# end

	def forward(self, tensorFirst, tensorSecond):
		return FunctionSepconv()(tensorFirst, tensorSecond)
		# end
		# end