CI_REGISTRY_IMAGE:=lilith-registry-vpc.cn-shanghai.cr.aliyuncs.com/tsh-hdp/aispeech
ci-build-buildah:
	echo "10.65.3.241 deb.debian.org" >> /etc/hosts
	echo "10.65.3.241 download.pytorch.org" >> /etc/hosts
	echo "10.65.3.241 developer.download.nvidia.com" >> /etc/hosts
	buildah bud --format=docker -f Dockerfile --layers --cache-to $(CI_REGISTRY_IMAGE)-cache --cache-from $(CI_REGISTRY_IMAGE)-cache -t $(CI_REGISTRY_IMAGE):${CI_COMMIT_SHA}  .
	# 推送镜像
	buildah push $(CI_REGISTRY_IMAGE):${CI_COMMIT_SHA} docker://$(CI_REGISTRY_IMAGE):${CI_COMMIT_SHA}
ci-deploy-app:
	export CI_REGISTRY_IMAGE=$(CI_REGISTRY_IMAGE) && j2 deploy.yaml > /tmp/app.yaml
	kubectl apply -n ${NAMESPACE} -f /tmp/app.yaml
