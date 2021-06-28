
if [ $# -ne 2 ];then
   echo "Usage:$0 (docker image) (repository name)" 
   exit 1
fi

docker_image=$1
repo_name=$2


repo_uri=`aws ecr create-repository --repository-name $repo_name --region ap-northeast-1 | jq '.repository' | jq '.repositoryUri' | sed 's/"//g'`
echo $repo_uri
sudo docker tag $docker_image $repo_uri

aws ecr get-login-password | docker login --username AWS --password-stdin $repo_uri

docker push $repo_uri
