#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>
using namespace InferenceEngine;
using namespace std;
using namespace cv;


double sigmoid(double x) {
	return (1 / (1 + exp(-x)));
}

vector<int> get_anchors(int net_grid) {
	vector<int> anchors(6);
	int anchor_80[6] = { 10,13, 16,30, 33,23 };
	int anchor_40[6] = { 30,61, 62,45, 59,119 };
	int anchor_20[6] = { 116,90, 156,198, 373,326 };
	if (net_grid == 80) { anchors.insert(anchors.begin(), anchor_80, anchor_80 + 6); }
	else if (net_grid == 40) { anchors.insert(anchors.begin(), anchor_40, anchor_40 + 6); }
	else if (net_grid == 20) { anchors.insert(anchors.begin(), anchor_20, anchor_20 + 6); }
	return anchors;
}

bool parse_yolov5(const Blob::Ptr &blob, int net_grid, float cof_threshold,
	vector<Rect>& o_rect, vector<float>& o_rect_cof) {
	vector<int> anchors = get_anchors(net_grid);
	LockedMemory<const void> blobMapped = as<MemoryBlob>(blob)->rmap();
	const float *output_blob = blobMapped.as<float *>();
	//n������n+5
	int item_size = 25;
	size_t anchor_n = 3;
	for (int n = 0; n < anchor_n; ++n)
		for (int i = 0; i < net_grid; ++i)
			for (int j = 0; j < net_grid; ++j)
			{
				double box_prob = output_blob[n*net_grid*net_grid*item_size + i * net_grid*item_size + j * item_size + 4];
				box_prob = sigmoid(box_prob);
				//�����ŶȲ��������������ŶȲ�����
				if (box_prob < cof_threshold)
					continue;

				//�����ĵ�����ת��Ϊ�ǵ�����
				double x = output_blob[n*net_grid*net_grid*item_size + i * net_grid*item_size + j * item_size + 0];
				double y = output_blob[n*net_grid*net_grid*item_size + i * net_grid*item_size + j * item_size + 1];
				double w = output_blob[n*net_grid*net_grid*item_size + i * net_grid*item_size + j * item_size + 2];
				double h = output_blob[n*net_grid*net_grid*item_size + i * net_grid*item_size + j * item_size + 3];
				double max_prob = 0;
				int idx = 0;
				for (int t = 5; t < 25; ++t) {
					double tp = output_blob[n*net_grid*net_grid*item_size + i * net_grid*item_size + j * item_size + t];
					tp = sigmoid(tp);
					if (tp > max_prob) {
						max_prob = tp;
						idx = t;
					}
				}
				float cof = box_prob * max_prob;
				//�޳��߿����Ŷ�С����ֵ�ı߿�
				if (cof < cof_threshold)
					continue;

				x = (sigmoid(x) * 2 - 0.5 + j)*640.0f / net_grid;
				y = (sigmoid(y) * 2 - 0.5 + i)*640.0f / net_grid;
				w = pow(sigmoid(w) * 2, 2) * anchors[n * 2];
				h = pow(sigmoid(h) * 2, 2) * anchors[n * 2 + 1];
				double r_x = x - w / 2;
				double r_y = y - h / 2;
				Rect rect = Rect(round(r_x), round(r_y), round(w), round(h));
				o_rect.push_back(rect);
				o_rect_cof.push_back(cof);
			}
	if (o_rect.size() == 0)
		return false;
	else
		return true;
}







int main() {
	// ��ʼ����������
	Core ie;
	// ��ȡת���õ���.xml��.bin�ļ�
	CNNNetwork network = ie.ReadNetwork("./openvino/yolov5s.xml", "./openvino/yolov5s.bin");
	// ��ȡ�������������ʽ
	// ��ģ���л�ȡ�������ݵĸ�ʽ��Ϣ	
	InputsDataMap inputsInfo = network.getInputsInfo();
	InputInfo::Ptr& input = inputsInfo.begin()->second;
	string inputs_name = inputsInfo.begin()->first;
	ICNNNetwork::InputShapes inputShapes = network.getInputShapes();
	network.reshape(inputShapes);

	// ��ģ���л�ȡ�ƶϽ���ĸ�ʽ
	OutputsDataMap outputsInfo = network.getOutputsInfo();
	vector<string> OutputsBlobs_names;
	for (auto& item_out : outputsInfo) {
		OutputsBlobs_names.push_back(item_out.first);
		item_out.second->setPrecision(Precision::FP32);
	}

	// ��ȡ��ִ������,�����CPUָ�����ƶ����е�����,��ѡ"GPU"
	ExecutableNetwork executable_network = ie.LoadNetwork(network, "CPU");

	// ��������
	InferRequest infer_request = executable_network.CreateInferRequest();
	InferenceEngine::Blob::Ptr lrInputBlob = infer_request.GetBlob(inputs_name);
	float*  buffer = lrInputBlob->buffer().as<float*>();
	Mat src = cv::imread("./img/test.jpg");
	size_t h = lrInputBlob->getTensorDesc().getDims()[2];
	size_t w = lrInputBlob->getTensorDesc().getDims()[3];
	size_t image_size = h * w;
	Mat inframe = src.clone();
	cv::resize(src, src, Size(640, 640));
	cv::cvtColor(src, src, COLOR_BGR2RGB);
	InferenceEngine::LockedMemory<void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(lrInputBlob)->wmap();
	float* blob_data = blobMapped.as<float*>();

	//nchw
	for (size_t row = 0; row < h; row++) {
		for (size_t col = 0; col < w; col++) {
			for (size_t ch = 0; ch < 3; ch++) {
				blob_data[image_size*ch + row * w + col] = float(src.at<Vec3b>(row, col)[ch]) / 255.0f;
			}
		}
	}

	//ִ������
	infer_request.Infer();
	//�������Ŷ���ֵ��NMS��ֵ
	float _cof_threshold = 0.1;
	float _nms_area_threshold = 0.5;
	//��ȡ������
	vector<Rect> origin_rect;
	vector<float> origin_rect_cof;
	int s[3] = { 80,40,20 };
	vector<Blob::Ptr> blobs;
	int i = 0;
	for (auto OutputsBlob_name : OutputsBlobs_names) {
		Blob::Ptr OutputBlob = infer_request.GetBlob(OutputsBlob_name);
		parse_yolov5(OutputBlob, s[i], _cof_threshold, origin_rect, origin_rect_cof);
		++i;
	}
	//���������ռ����
	vector<int> final_id;
	//����NMS���������ص�����ĺ�ѡ��
	dnn::NMSBoxes(origin_rect, origin_rect_cof, _cof_threshold, _nms_area_threshold, final_id);
	//����final_id��ȡ���ս��
	for (size_t i = 0; i < final_id.size(); ++i)
	{
		int idx = final_id[i];
		Rect box = origin_rect[idx];
		cv::rectangle(inframe, box, Scalar(140, 199, 0), 1, 8, 0);
	}

	cv::imwrite("./img/output.jpg", inframe);
}

